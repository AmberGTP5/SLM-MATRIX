import os
import re
import json
import argparse
import requests
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger


API_BASE = "https://api.together.xyz/v1"
API_KEY = 
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

class ConsistencyChecker:
    def __init__(self, max_tokens=2048, temperature=0.0):
        self.api_key = API_KEY
        self.api_base = API_BASE
        self.model_name = MODEL_NAME
        self.max_tokens = max_tokens # For regeneration
        self.regeneration_temperature = temperature # For regeneration
        self.name_judgment_temperature = 0.1 # For LLM name judgment
        self.name_judgment_max_tokens = 128  # Max tokens for simple Yes/No + reason

    def call_llm_api(self, prompt: str, temperature: float, max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            # logger.debug(f"LLM API Call. Model: {self.model_name}, Temp: {temperature}, MaxTokens: {max_tokens}")
            # logger.debug(f"LLM Prompt (first 200 chars): {prompt[:200]}...")
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # logger.debug(f"LLM Response (first 100 chars): {content[:100]}...")
            return content
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"API HTTP error occurred: {http_err} - {response.text}")
            return f"API call failed: HTTP error {response.status_code}"
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return "API call failed, cannot get response."

    def extract_trajectories_from_file(self, filepath: str) -> List[Dict[str, str]]:
        
        trajectories_data: List[Dict[str, str]] = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            trajectory_blocks = re.finditer(
                r"(?s)## Candidate Trajectory \d+:\s*Action Sequence: (.*?)\s*Visits: (\d+), Value: ([\d\.]+)\s*\n\n"
                r"\*\*Original Query\*\*:([\s\S]*?)"
                r"(?=\n\n\*\*Step 1|\n\n\*\*Final Structured Output\*\*)"
                r"([\s\S]*?)"
                r"\*\*Final Structured Output\*\*:\s*([\s\S]*?)"
                r"(?=## Candidate Trajectory \d+:|$)",
                content
            )
            for match_idx, match in enumerate(trajectory_blocks):
                action_sequence = match.group(1).strip()
                original_query = match.group(4).strip()
                mcts_steps_analysis_block = match.group(5).strip()
                original_final_output = match.group(6).strip()
                last_step_content_for_regen = "No analysis steps available."
                if mcts_steps_analysis_block:
                    step_block_pattern = r"(\*\*Step \d+ \[A\d+\]\*\*:.*?\n[\s\S]*?)(?=\n\n\*\*Step \d+ \[A\d+\]\*\*|\Z)"
                    all_step_blocks = re.findall(step_block_pattern, mcts_steps_analysis_block, re.DOTALL)
                    if all_step_blocks:
                        last_step_content_for_regen = all_step_blocks[-1].strip()
                    else:
                        logger.warning(f"Trajectory {match_idx+1} (Action Sequence: {action_sequence}): No formatted steps found in the step analysis block. Using the entire step analysis block as context. Content fragment: '{mcts_steps_analysis_block[:100]}...'")
                        last_step_content_for_regen = mcts_steps_analysis_block
                else:
                     logger.warning(f"Trajectory {match_idx+1} (Action Sequence: {action_sequence}): Step analysis content is empty.")
                trajectories_data.append({
                    "original_query": original_query,
                    "analysis_context_for_regen": last_step_content_for_regen,
                    "original_final_output": original_final_output
                })
            if not trajectories_data: logger.warning(f"Failed to extract any formatted trajectories from file {filepath}.")
        except FileNotFoundError: logger.error(f"Error: trajectory file '{filepath}' not found.")
        except Exception as e: logger.error(f"Error occurred while extracting trajectories: {e}")
        return trajectories_data

    def _parse_quantity_string(self, quantity_full_string: str) -> Tuple[Optional[str], Optional[str]]:
        
        quantity_full_string = quantity_full_string.strip()
        match = re.match(r"([-+]?\d*\.?\d+(?:\s*±\s*[-+]?\d*\.?\d+)?)\s*(.*)", quantity_full_string)
        value_str, unit_str = None, None
        if match:
            value_str, unit_str = match.group(1).strip(), match.group(2).strip()
            if not unit_str or unit_str == '-': unit_str = None
        elif re.fullmatch(r"[-+]?\d*\.?\d+", quantity_full_string):
            value_str = quantity_full_string
        return value_str, unit_str

    def extract_material_triplets(self, output_text: str) -> List[Dict[str, Any]]:
        
        triplets: List[Dict[str, Any]] = []
        if not output_text or not isinstance(output_text, str) or not output_text.strip(): return triplets
        item_pattern = re.compile(r"-\s*Material name:\s*(.*?)\s*\n-\s*Quantity:\s*(.*)", re.MULTILINE)
        matches = item_pattern.findall(output_text)
        for mat_name_match, quantity_full_str_match in matches:
            material_name = mat_name_match.strip()
            quantity_full_string = quantity_full_str_match.strip()
            value_str, unit_str = self._parse_quantity_string(quantity_full_string)
            if value_str is not None:
                triplets.append({"material": material_name, "property": "", "value": value_str, "unit": unit_str})
            else: logger.warning(f"Skipping triplet for '{material_name}' due to unparsable quantity: '{quantity_full_string}'")
        if not triplets and "Material Information Summary:" in output_text and len(output_text.replace("Material Information Summary:", "").strip()) >0:
             logger.warning(f"Failed to extract any triples from the following output (extract_material_triplets):\n{output_text[:300]}...")
        return triplets

    def _normalize_value_for_comparison(self, value_str: Optional[str]) -> Optional[str]:
        
        if value_str is None: return None
        val_to_compare = str(value_str).strip()
        if "±" in val_to_compare:
            parts = [p.strip() for p in val_to_compare.split("±")]
            if len(parts) == 2:
                try:
                    main_val_norm = str(float(parts[0])) if re.fullmatch(r"[-+]?\d*\.?\d+", parts[0]) else parts[0]
                    error_val_norm = str(float(parts[1])) if re.fullmatch(r"[-+]?\d*\.?\d+", parts[1]) else parts[1]
                    return f"{main_val_norm} ± {error_val_norm}"
                except ValueError: return val_to_compare
            else: return val_to_compare
        try:
            if re.fullmatch(r"[-+]?\d*\.?\d+", val_to_compare): return str(float(val_to_compare))
        except ValueError: pass
        return val_to_compare

    def _are_names_semantically_equivalent(self, name_a: str, name_b: str, original_query_text: str) -> bool:
        """
        Use LLM to judge whether two material/parameter names refer to the same entity, considering the context provided by an original query text.
        """
        prompt = f"""Your task is to determine if two given material or parameter names refer to the same entity, considering the context provided by an original query text.

Original Query Text (for context):
\"\"\"
{original_query_text}
\"\"\"

Name A: "{name_a}"
Name B: "{name_b}"

Based on the Original Query Text and general knowledge, do "Name A" and "Name B" refer to substantially the same material, chemical, process parameter, or property?

Your answer MUST be a single word: "Yes" or "No".
Then, on a new line, you MAY provide a very brief explanation (optional, max 1-2 sentences).

Example 1:
Original Query Text: "...BLF powder was used... then calcined at 800 degC..."
Name A: "BLF"
Name B: "BLF powder"
Judgment:
Yes
Explanation: Both refer to the BLF material, "BLF powder" is just more specific.

Example 2:
Original Query Text: "...an aqueous solution of malic acid (50 mL)..."
Name A: "aqueous solution of malic acid"
Name B: "malic acid solution volume"
Judgment:
Yes
Explanation: Both refer to the quantified malic acid solution.

Example 3:
Original Query Text: "...drying temperature was 120 degC... calcination temperature was 700 degC..."
Name A: "drying temperature"
Name B: "calcination temperature"
Judgment:
No
Explanation: These refer to distinct process parameters.

Now, make your judgment for the provided Name A and Name B:
Judgment:
"""
        logger.debug(f"LLM Name Judgment: Comparing '{name_a}' and '{name_b}'")
        response = self.call_llm_api(prompt, temperature=self.name_judgment_temperature, max_tokens=self.name_judgment_max_tokens)

        if "API call failed" in response or "API call failed" in response:
            logger.error(f"LLM name judgment API call failed for '{name_a}' vs '{name_b}'. Response: {response}")
            return False # Default to not equivalent on API failure to be conservative

        first_line = response.strip().split('\n')[0].strip().lower()
        # More robust check for "yes"
        if first_line.startswith("yes"):
            logger.debug(f"LLM judged '{name_a}' and '{name_b}' as SEMANTICALLY EQUIVALENT. Reason (if any): {response.strip().splitlines()[1:]}")
            return True
        else:
            logger.debug(f"LLM judged '{name_a}' and '{name_b}' as NOT semantically equivalent. Reason: {response}")
            return False

    def compare_triplets(self,
                         original_triplets: List[Dict[str, Any]],
                         new_triplets: List[Dict[str, Any]],
                         original_query_text: str,
                         # original_structured_output_text and new_structured_output_text are no longer needed here
                         ) -> Tuple[bool, str]:
        """
        Compare two sets of triples.
        1. The quantities must be strictly consistent.
        2. For each original triple, one and only one corresponding item must be found in the new triple.
        3. The values ​​and units of the corresponding items must be strictly consistent.
        4. The material names of the corresponding items, if different, call LLM to determine whether they are semantically equivalent.
        """
        num_original = len(original_triplets)
        num_new = len(new_triplets)

        if num_original != num_new:
            return False, f"Triple number mismatch: original version has {num_original} items, new version has {num_new} items."

        if num_original == 0: # Both are empty
            return True, "Both versions have no triples, considered consistent."

        # Convert to a more usable format for matching (value and unit normalized)
        # We will keep original names for LLM comparison if needed
        def process_triplets_for_matching(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            processed = []
            for t in triplets:
                norm_val = self._normalize_value_for_comparison(t.get("value"))
                norm_unit = str(t.get("unit")).strip().lower() if t.get("unit") and str(t.get("unit")).strip() else "none"
                processed.append({
                    "original_name": t.get("material"), # Keep original name
                    "name_lower": str(t.get("material")).strip().lower(), # For initial matching attempts
                    "value": norm_val,
                    "unit": norm_unit,
                    "matched_to_in_other_list": False # Flag to track matching
                })
            return processed

        processed_original = process_triplets_for_matching(original_triplets)
        processed_new = process_triplets_for_matching(new_triplets)
        
        mismatched_details = []
        all_match_found_for_originals = True

        for orig_idx, orig_triplet in enumerate(processed_original):
            found_match_for_this_orig = False
            # Try to find a match in new_triplets that hasn't been matched yet
            for new_idx, new_triplet in enumerate(processed_new):
                if new_triplet["matched_to_in_other_list"]:
                    continue # This new_triplet is already matched to a different original_triplet

                # Criteria 1: Value and Unit must be strictly identical
                if orig_triplet["value"] == new_triplet["value"] and \
                   orig_triplet["unit"] == new_triplet["unit"]:
                    
                    # Criteria 2: Names must be semantically equivalent
                    # First, check for exact (case-insensitive) name match for efficiency
                    if orig_triplet["name_lower"] == new_triplet["name_lower"]:
                        orig_triplet["matched_to_in_other_list"] = True
                        new_triplet["matched_to_in_other_list"] = True
                        found_match_for_this_orig = True
                        logger.debug(f"Exact match found: '{orig_triplet['original_name']}' ({orig_triplet['value']} {orig_triplet['unit']})")
                        break # Move to the next original triplet
                    else:
                        # If names are different, call LLM to judge semantic equivalence
                        logger.info(f"Value/unit match, but names are different. Call LLM to judge semantic equivalence: "
                                    f"A='{orig_triplet['original_name']}', B='{new_triplet['original_name']}'")
                        if self._are_names_semantically_equivalent(
                            orig_triplet["original_name"], 
                            new_triplet["original_name"], 
                            original_query_text
                        ):
                            orig_triplet["matched_to_in_other_list"] = True
                            new_triplet["matched_to_in_other_list"] = True
                            found_match_for_this_orig = True
                            logger.info(f"LLM confirms semantic equivalence of names: '{orig_triplet['original_name']}' and '{new_triplet['original_name']}'")
                            break # Move to the next original triplet
                        else:
                            # LLM says names are not equivalent, this new_triplet is not a match for current orig_triplet
                            logger.info(f"LLM determines that the names are not equivalent: '{orig_triplet['original_name']}' and '{new_triplet['original_name']}'")
                            # Continue searching for other potential new_triplets for this orig_triplet
            
            if not found_match_for_this_orig:
                all_match_found_for_originals = False
                mismatched_details.append(
                    f"Item '{orig_triplet['original_name']}' ({orig_triplet['value']} {orig_triplet['unit']}) "
                    f"in the original output was not found in the new output with a value/unit match and semantically equivalent name."
                )
                # No need to break the outer loop here, we want to find all missing originals
        
        # After checking all original triplets, verify if all new triplets were also matched
        # This ensures a one-to-one mapping if counts were equal.
        # If all_match_found_for_originals is True, this check confirms no new_triplets were left unmatched.
        unmatched_new_count = sum(1 for t in processed_new if not t["matched_to_in_other_list"])
        if all_match_found_for_originals and unmatched_new_count > 0 :
             # This case should not happen if num_original == num_new and all originals were matched one-to-one.
             # It implies a logic error or a many-to-one match scenario not handled by current flags.
             # For strict one-to-one when counts are equal, if all originals found a unique match, all news must also be matched.
             all_match_found_for_originals = False # Treat as inconsistent
             mismatched_details.append(f"Logical error or mismatch: although all original items were found to match, there are still {unmatched_new_count} new items that were not matched.")


        if all_match_found_for_originals: # This implies unmatched_new_count is also 0 given the previous check
            return True, "All triples found semantic consistency (name flexible, value/unit strictly) in the corresponding items, and the number is consistent."
        else:
            return False, "One or more triples were not found to have semantic consistency in the corresponding items, or there is a matching logic problem.\nDetails:\n" + "\n".join(mismatched_details)

    def generate_new_output(self, analysis_context: str, original_query: str) -> str:
        
        prompt = f"""Synthesize the provided analysis to extract "entity name - numerical value - unit" triplets.
The goal is to capture any clearly named entity in the text that is directly quantified with a numerical value and a unit. This can include chemical substances, solutions, mixtures, and also explicitly quantified process parameters (like temperature, time, pH, ratios).

The original query was: "{original_query}"
The final analysis from the reasoning process (this candidate trajectory) is:
---
{analysis_context}
---

**Key Extraction & Formatting Rules:**
1.  **Extraction Target**:
    * Identify any entity (material, parameter, etc.) that is explicitly named and directly associated with a numerical value and a unit in the provided "final analysis".
    * If the analysis mentions components within a mixture and quantifies them separately, extract them.
2.  **Entity Naming**:
    * Use the most descriptive name for the entity as suggested by the analysis (e.g., "aqueous malic acid solution", "drying temperature", "pH of mixed solution").
    * The 'Material name' (or 'Entity name') field should accurately reflect what the quantity refers to. It MUST NOT contain the numerical value or unit of its quantity.
    * **Chemical Formulas**: If a chemical formula (e.g., Ba0.95La0.05FeO3-未) is mentioned, only extract it if the *entire compound itself* is given an explicit external quantity (e.g., "10 grams of Ba0.95La0.05FeO3-未"). Do NOT extract subscript numbers within a formula as if they are quantities of that material.
3.  **Quantities**:
    * Record the precise numerical value and its corresponding unit as found in the analysis.
    * The 'Quantity' field is the ONLY place for the numerical value and its unit.
    * Include error margins (e.g., ±0.1) if present.
4.  **Focus on Accuracy and Textual Grounding**:
    * Ensure all extracted triplets (name, value, unit) are directly supported by the provided "final analysis" content.
    * Avoid inventing information or misattributing quantities.
    * If a numerical value is mentioned but it's clearly a citation number (e.g., "[28]") or an unassociated identifier, do NOT treat it as a quantity for a material/parameter unless the analysis explicitly and reasonably links them.
5.  **Output Structure (FOLLOW THIS EXACTLY):**
    * Begin the entire output with "Material Information Summary:".
    * For each distinct extracted triplet:
        - Material name: [Name of the Entity/Material/Parameter]
        - Quantity: [Numerical Value][±Error if any] [Unit]  **<-- Unit is ONLY part of this line.**
    * **ABSOLUTELY NO separate 'Unit:' line.**
    * Use one empty line between different "Material Information Summary:" blocks.
    * NO markdown formatting (like bold ** or italics *) in the "Material Information Summary:" section. Use a dash (-) for list items as shown.
6.  **Multiple Quantifications for One Entity**:
    * If a single entity has multiple distinct quantifications (e.g., a solution with both volume and concentration mentioned), create SEPARATE "Material Information Summary:" entries for EACH quantification, repeating the entity name.

**Examples (ADHERE STRICTLY TO THIS FORMAT AND NAMING):**
(Example: analysis mentions "Ba (9.5 mmol) acetates")
Material Information Summary:
- Material name: Ba acetates
- Quantity: 9.5 mmol
(Example: analysis mentions "drying was done at 120 degC for 2 h")
Material Information Summary:
- Material name: drying temperature
- Quantity: 120 degC
Material Information Summary:
- Material name: drying time
- Quantity: 2 h
(Example: analysis mentions "the pH was 6")
Material Information Summary:
- Material name: pH
- Quantity: 6
(Example: analysis mentions "The compound Ba0.95La0.05FeO3-未 was used." -> DO NOT EXTRACT if not externally quantified. If it says "5 grams of Ba0.95La0.05FeO3-未", then extract:
Material Information Summary:
- Material name: Ba0.95La0.05FeO3-未
- Quantity: 5 grams)

Please extract all relevant "entity name - numerical value - unit" triplets from the provided "final analysis" and structure them now according to all the rules above.
"""
        return self.call_llm_api(prompt, temperature=self.regeneration_temperature, max_tokens=self.max_tokens)

    def check_consistency(self, trajectory_file_path: str):
        trajectories_data = self.extract_trajectories_from_file(trajectory_file_path)
        results = []
        if not trajectories_data:
            logger.warning(f"Failed to extract any trajectories from file {trajectory_file_path}.")
            return results

        for i, traj_info in enumerate(trajectories_data):
            original_query_text = traj_info.get("original_query", "Original query content missing")
            analysis_context_for_regen = traj_info.get("analysis_context_for_regen", "Analysis context missing")
            original_final_output_text = traj_info.get("original_final_output", "")
            
            trajectory_result = {
                "trajectory_id": i + 1,
                "original_query": original_query_text,
                "analysis_context_for_regen": analysis_context_for_regen,
                "original_output": original_final_output_text,
                "new_output": "",
                "is_consistent": False,
                "comparison_details": "Failed to start comparison"
            }
            
            if analysis_context_for_regen == "Analysis context missing" or \
               original_final_output_text.strip() == "" or \
               analysis_context_for_regen == "No available analysis steps.":
                trajectory_result["comparison_details"] = "Analysis context or original output incomplete/missing."
                results.append(trajectory_result)
                continue

            logger.info(f"Trajectory {i+1}: Regenerating output...")
            new_generated_output_text = self.generate_new_output(analysis_context_for_regen, original_query_text)
            trajectory_result["new_output"] = new_generated_output_text.strip() if new_generated_output_text else ""
            
            if "API call failed" in trajectory_result["new_output"] or "API call failed" in trajectory_result["new_output"]:
                trajectory_result["comparison_details"] = "New output generation failed (API problem)."
            elif not trajectory_result["new_output"]:
                 trajectory_result["comparison_details"] = "New output is empty."
            else:
                original_triplets = self.extract_material_triplets(original_final_output_text)
                new_triplets = self.extract_material_triplets(trajectory_result["new_output"])
                
                is_consistent, comparison_details = self.compare_triplets(
                    original_triplets, 
                    new_triplets,
                    original_query_text
                )
                trajectory_result["is_consistent"] = is_consistent
                trajectory_result["comparison_details"] = comparison_details
            
            results.append(trajectory_result)
        return results
            
    def generate_report(self, results: List[Dict[str, Any]], trajectory_file: str) -> str:
        
        report_lines = [f"# Material extraction consistency check report ({os.path.basename(trajectory_file)})\n"]
        total = len(results)
        if total == 0:
            report_lines.append("No trajectories processed.")
            return "\n".join(report_lines)
        consistent_count = sum(1 for r in results if r["is_consistent"])
        report_lines.append(f"## Overall statistics\n")
        report_lines.append(f"- Total number of trajectories checked: {total}")
        report_lines.append(f"- Number of consistent trajectories: {consistent_count}")
        report_lines.append(f"- Number of inconsistent trajectories: {total - consistent_count}")
        valid_to_check_count = sum(1 for r in results if r["comparison_details"] not in [
            "Analysis context or original output incomplete/missing.",
            "New output generation failed (API problem).",
            "New output is empty.",
            "Failed to start comparison"
        ])
        if valid_to_check_count > 0:
            report_lines.append(f"- The consistency ratio of valid checks: {consistent_count/valid_to_check_count*100:.2f}% ({consistent_count}/{valid_to_check_count})")
        else:
            report_lines.append(f"- The consistency ratio of valid checks: N/A (no valid trajectories for comparison)")
        report_lines.append("")
        report_lines.append("## Detailed results\n")
        for r in results:
            report_lines.append(f"### Trajectory {r['trajectory_id']}")
            report_lines.append(f"- **Consistency**: {'✓ Consistent' if r['is_consistent'] else '✗ Inconsistent'}")
            report_lines.append(f"- **Comparison details**: {r['comparison_details']}") # This will now contain reasons from the new compare_triplets
            report_lines.append(f"\n<details><summary>View input analysis, original output, and regenerated output</summary>\n")
            report_lines.append(f"**Original query (Original Query for LLM Judgment)**:\n```text\n{r.get('original_query', 'N/A')}\n```")
            report_lines.append(f"**Analysis context for regeneration (Analysis Context for Regeneration)**:\n```text\n{r['analysis_context_for_regen']}\n```")
            report_lines.append(f"**Original output (Original Output)**:\n```text\n{r['original_output']}\n```")
            report_lines.append(f"**Regenerated output (Regenerated Output)**:\n```text\n{r['new_output']}\n```\n</details>\n")
        return "\n".join(report_lines)

def main():
    
    parser = argparse.ArgumentParser(description='Material extraction consistency checker V3 (LLM Name Check)')
    parser.add_argument('--file', type=str, default="material_extraction_trajectories.txt", 
                        help='Path to the file containing MCTS trajectories')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature parameter (for regeneration)')
    parser.add_argument('--name_judgment_temperature', type=float, default=0.1, # Default temperature for name judgment
                        help='LLM temperature parameter (for name semantic equivalence judgment)')
    parser.add_argument('--name_judgment_max_tokens', type=int, default=128,
                        help='LLM maximum output token number (for name semantic equivalence judgment)')                     
    parser.add_argument('--max_tokens', type=int, default=1024, 
                        help='LLM maximum output token number (for regeneration)')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f'LLM model name for verification (default: {MODEL_NAME})')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging')
    
    args = parser.parse_args()
    
    if args.debug:
        import sys
        logger.remove() 
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("DEBUG mode enabled.")
    else:
        import sys
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    checker = ConsistencyChecker(max_tokens=args.max_tokens, temperature=args.temperature)
    checker.name_judgment_temperature = args.name_judgment_temperature
    checker.name_judgment_max_tokens = args.name_judgment_max_tokens

    if args.model_name != MODEL_NAME:
        checker.model_name = args.model_name
        logger.info(f"Using model specified in command line: {checker.model_name}")
    
    if not os.path.exists(args.file):
        logger.error(f"Error: file '{args.file}' does not exist.")
        return
    
    logger.info(f"Starting to check trajectory file: {args.file}, LLM model: {checker.model_name}, regeneration temperature: {args.temperature}, name judgment temperature: {checker.name_judgment_temperature}")
    results = checker.check_consistency(args.file)
    
    if results:
        report_content = checker.generate_report(results, args.file)
        base_filename = os.path.splitext(os.path.basename(args.file))[0]
        report_filename = f"consistency_report_{base_filename}_v3_name_judged.md"
        
        try:
            with open(report_filename, "w", encoding="utf-8") as f: f.write(report_content)
            logger.info(f"Consistency check completed. Report saved to: {report_filename}")
            print(f"\nConsistency check completed. Report saved to: {report_filename}")
            total_results, consistent_count, effectively_checked_count = len(results), sum(1 for r in results if r["is_consistent"]), 0
            for r_val in results:
                if r_val["comparison_details"] not in ["Analysis context or original output incomplete/missing.", "New output generation failed (API problem).", "New output is empty.", "Failed to start comparison"]: effectively_checked_count += 1
            if effectively_checked_count > 0: print(f"Check summary: {consistent_count}/{effectively_checked_count} valid trajectories consistent ({consistent_count/effectively_checked_count*100:.2f}%)")
            elif total_results > 0: print(f"Checked {total_results} trajectory entries, but none could be effectively compared for consistency.")
            else: print("No trajectories were effectively processed for summary statistics.")
        except IOError as e: logger.error(f"Failed to save report file: {e}")
    else: logger.info("No valid trajectories were extracted from the file, or there was an error during processing, so no report was generated.")

if __name__ == "__main__":
    main()