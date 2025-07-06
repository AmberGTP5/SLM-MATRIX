import numpy as np
import math
import random
import os
from typing import List, Dict, Any, Tuple, Optional
import copy
import time
import requests
import json
import argparse
from loguru import logger
import re


API_BASE = "https://api.together.xyz/v1"
API_KEY = 
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

class MCTSNode:
    def __init__(self, state: str, parent=None, action_type=None, action_content=None):
        self.state = state  
        self.parent = parent  
        self.action_type = action_type  
        self.action_content = action_content  
        self.children = []  
        self.visits = 0  
        self.value = 0  
        self.is_terminal = False  
        self.terminal_value = None  
        
    def is_fully_expanded(self):
        """Check if the node is fully expanded (tried all 5 action types)"""
        action_types = set(child.action_type for child in self.children)
        return len(action_types) >= 5
    
    def is_leaf(self):
        """Check if the node is a leaf node"""
        return len(self.children) == 0
    
    def get_value(self):
        """Get the value of the node"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits
    
    def get_trajectory(self):
        """Get the complete trajectory from the root node to the current node"""
        if self.parent is None:
            return [{"state": self.state, "action_type": None, "action_content": None}]
        
        parent_trajectory = self.parent.get_trajectory()
        parent_trajectory.append({
            "state": self.state, 
            "action_type": self.action_type,
            "action_content": self.action_content
        })
        return parent_trajectory
    
    def get_formatted_trajectory(self):
        """Get the formatted trajectory for display"""
        trajectory = self.get_trajectory()
        formatted = []
        for i, step in enumerate(trajectory):
            if i == 0:  # root node
                formatted.append(f"Query: {step['state']}")
            else:
                formatted.append(f"Step {i}: [Action {step['action_type']}] {step['state']}")
        return "\n\n".join(formatted)
    
    def get_action_sequence(self):
        """Get the action sequence"""
        trajectory = self.get_trajectory()
        return [step["action_type"] for step in trajectory if step["action_type"]]

class MaterialDataExtractor:
    def __init__(self, max_tokens=2048, temperature=0.7):
        
        self.api_key = API_KEY
        self.api_base = API_BASE
        self.model_name = MODEL_NAME
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # MCTS configuration
        self.c_puct = 1.5  # UCT exploration constant
        
        # Action space design - 5 action types
        # Please refer to the thesis prompts for action definitions
        self.action_templates = {
            "A1": {
                "name": "Quick thinking and material recognition",
                "prompt": "Let's quickly think about the materials mentioned in this literature. First identify the key material names:"
            },
            "A2": {
                "name": "Complete reasoning and property extraction",
                "prompt": "Based on the identified material information, let me complete the extraction of its properties. Extract all related numerical values and units from the text:"
            },
            "A3": {
                "name": "Problem decomposition and sub-problem exploration",
                "prompt": "Let me decompose this complex problem into simpler sub-problems:\n1. What materials are mentioned in the text?\n2. What are the properties of each material?\n3. What are the numerical values and units of these properties?\nI'll answer the first sub-problem first:"
            },
            "A4": {
                "name": "Re-evaluation for accuracy improvement",
                "prompt": "Let me re-evaluate my answer using the chain of thought method. Think step by step:\n1. What materials are explicitly mentioned in the text?\n2. For each material, can I find the exact property values?\n3. What are the units of these values?"
            },
            "A5": {
                "name": "Problem reconstruction and result verification",
                "prompt": "Let me reconsider this problem and my answer. The problem requires extracting triplets of material names, numerical values, and units. I need to ensure no information is missed. Let me rephrase and verify:"
            }
        }
    
    def print_action_space(self):
        """Print the action space definition"""
        print("\n===== Action space definition =====")
        for action_id, action_info in self.action_templates.items():
            print(f"{action_id}: {action_info['name']}")
            print(f"Prompt template: {action_info['prompt']}\n")
    
    def call_llm_api(self, prompt, max_tokens=None, temperature=None):
        """Call the LLM API"""
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
            
        try:
            # Build the message format
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare the API request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Set headers
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Send the API request
            response = requests.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
            
            # Parse the response
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Error: No content in API response"
            else:
                return f"Error: API returned status code {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"API call failed: {str(e)}"
    
    def select(self, node):
        """Selection phase: select the most promising child node from the current node"""
        # If the node is a leaf node or terminal node, return directly
        if node.is_leaf() or node.is_terminal:
            return node
            
        # If the node is not fully expanded, return the node for expansion
        if not node.is_fully_expanded():
            return node
            
        # Use the UCT formula to select the best child node
        def uct_score(child):
            # Exploitation term: known node value
            exploitation = child.get_value()
            # Exploration term: encourage nodes with fewer visits
            exploration = self.c_puct * math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration
            
        # Select the child node with the highest UCT score
        selected_child = max(node.children, key=uct_score)
        
        # Recursive selection
        return self.select(selected_child)
    
    def expand(self, node):
        """Expansion phase: add a new child node to the current node"""
        # Get the action types that have not been tried
        tried_actions = set(child.action_type for child in node.children)
        available_actions = [f"A{i}" for i in range(1, 6) if f"A{i}" not in tried_actions]
        
        if not available_actions:  # If all actions have been tried
            return node
            
        # Randomly select an action type that has not been tried
        action_type = random.choice(available_actions)
        
        # Generate different prompts based on the action type
        action_template = self.action_templates[action_type]["prompt"]
        
        # Build the prompt, including the previous context
        context = node.state
        prompt = f"{context}\n\n{action_template}"
        
        # Call the LLM API to get the response
        response = self.call_llm_api(prompt)
        
        # Create a new node
        new_node = MCTSNode(
            state=response,
            parent=node,
            action_type=action_type,
            action_content=action_template
        )
        
        # Add the new node as a child node
        node.children.append(new_node)
        
        return new_node
    
    def simulate(self, node):
        """Simulation phase: evaluate the value of the current node"""
        # Extract the final output of the trajectory
        final_output = node.state
        
        # Evaluate the quality of the final output directly
        evaluation_prompt = f"""
        The original query aims to extract structured "entity name - numerical value - unit" triplets from scientific text.
        An entity can be a chemical substance, solution, mixture, or even a clearly quantified process parameter (like temperature, time) if it's explicitly named and associated with a numerical value AND AN EXPLICIT UNIT in the text.
        **Triplets without an explicit unit (e.g., pH 6, ratio 1.5) are NOT considered valid targets for extraction.**

        Current analysis content from the reasoning path:
        ---
        {final_output}
        ---

        Evaluate this "Current analysis content". Your primary goal is to determine if this analysis clearly identifies potential "entity name - numerical value - EXPLICIT unit" triplets that are directly supported by the text.

        Consider the following:
        * **Identification Clarity**: Does the analysis clearly point to specific names, values, AND THEIR EXPLICIT UNITS? If a unit is missing for a quantified entity, it's a flaw.
        * **Textual Support**: Is the identified information (including the unit) directly and accurately supported by the original text? Avoid hallucination.
        * **Completeness (Broad Sense)**: Does the analysis seem to be progressing towards identifying multiple such valid (name-value-unit) triplets if present, or is it stuck/irrelevant/identifying unit-less quantities?
        * **Avoid Gross Errors**: While broader, avoid obvious misinterpretations (e.g., using a citation number as a quantity, misattributing quantities, or suggesting extraction of entities that lack explicit units).

        A score near 1.0 indicates the analysis is excellent at identifying multiple, clear, textually-supported triplets (name, value, AND explicit unit).
        A score near 0.0 indicates the analysis is unhelpful, contains gross misinterpretations, fails to identify any clear triplets, or focuses on unit-less quantities.

        Comprehensive Score (0.0 - 1.0): [Provide a float number reflecting the quality for valid triplet extraction]
        Brief Rationale (optional):
        """
        
        # Call the API to get the evaluation
        evaluation = self.call_llm_api(evaluation_prompt)
        
        # Extract the score from the evaluation
        try:
            # Try to extract the score from the evaluation response, compatible with "Comprehensive Score: X.X" and "Comprehensive Score (0.0 - 1.0): X.X" format
            score_match = re.search(r"Comprehensive Score(?:\s*\(0\.0 - 1\.0\))?:\s*([\d\.]+)", evaluation, re.IGNORECASE)
            if score_match:
                score_str = score_match.group(1)
                score = float(score_str)
                # Ensure the score is between 0.0 and 1.0
                logger.info(f"Successfully parsed the standard format score: {score:.2f}. Evaluation content snippet: {evaluation[:150]}...")
                return max(0.0, min(1.0, score))
            else:
                
                logger.warning(f"Cannot parse the 'Comprehensive Score: X.X' or 'Comprehensive Score (0.0 - 1.0): X.X' format from the evaluation response. Evaluation content: {evaluation[:500]}...")
                
                # Alternative logic: check if DEFECTS are explicitly mentioned
                if "DEFECT 1" in evaluation.upper() or \
                   "DEFECT 2" in evaluation.upper() or \
                   "DEFECT 3" in evaluation.upper() or \
                   "DEFECT 4" in evaluation.upper() or \
                   "DEFECT 5" in evaluation.upper(): # Convert to uppercase to match the DEFECT in the prompt
                    logger.info("Alternative logic: detected explicit defects in the evaluation, giving a score of 0.2.")
                    return 0.2 
                
                logger.info("Alternative logic: no score parsed and no explicit defects detected, giving a default penalty score of 0.35.")
                return 0.35 

        except Exception as e:
            logger.error(f"Error parsing the evaluation score: {e} - Evaluation content: {evaluation[:500]}...")
            return 0.2
    
    def backpropagate(self, node, reward):
        """Backpropagation phase: update the statistics of the current node and its ancestors"""
        # Update the current node
        node.visits += 1
        node.value += reward
        
        # Recursively update the ancestor nodes
        if node.parent:
            self.backpropagate(node.parent, reward)
    
    def run_mcts(self, query, num_simulations=25, max_depth=5):
        """Run the MCTS, generate the reasoning trajectory"""
        root = MCTSNode(query)
        
        for i in range(num_simulations):
            print(f"Running simulation {i+1}/{num_simulations}...")
            
            # 1. Selection
            selected_node = self.select(root)
            
            # 2. Expansion
            if not selected_node.is_terminal and len(selected_node.get_trajectory()) <= max_depth:
                new_node = self.expand(selected_node)
                
                # 3. Simulation
                reward = self.simulate(new_node)
                
                # 4. Backpropagation
                self.backpropagate(new_node, reward)
        
        # Collect the best trajectories and search tree information
        best_trajectories = self.get_best_trajectories(root, top_n=3)
        search_tree = self.generate_tree_visualization(root)
        
        return best_trajectories, root, search_tree
    
    def generate_tree_visualization(self, root, max_depth=3):
        """Generate the text visualization of the search tree"""
        def _generate_tree(node, depth=0, prefix="", is_last=True, max_depth=3):
            if depth > max_depth:
                return ""
                
            # The text display of the current node
            if depth == 0:
                node_text = f"{node.state[:50]}..." if len(node.state) > 50 else node.state
            else:
                node_text = f"[{node.action_type}] {node.state[:50]}..." if len(node.state) > 50 else node.state
                node_text += f" (Visits: {node.visits}, Value: {node.get_value():.2f})"
            
            # Build the current line
            line = prefix + ("└── " if is_last else "├── ") + node_text + "\n"
            
            # Build the prefix of the child nodes
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Recursively process the child nodes
            for i, child in enumerate(sorted(node.children, key=lambda c: c.visits, reverse=True)):
                is_last_child = (i == len(node.children) - 1)
                line += _generate_tree(child, depth + 1, new_prefix, is_last_child, max_depth)
                
            return line
            
        return _generate_tree(root, max_depth=max_depth)
    
    def get_best_trajectories(self, root, top_n=3):
        """Get the best reasoning trajectories"""
        # Auxiliary function: find the best leaf node under the node
        def get_best_leaf(node):
            if node.is_terminal or node.is_leaf():
                return node
                
            best_child = max(node.children, key=lambda c: c.visits)
            return get_best_leaf(best_child)
            
        # Group the best trajectories by action type
        best_trajectories = []
        action_branches = {}
        
        # Group the child nodes by action type
        for child in root.children:
            if child.action_type not in action_branches:
                action_branches[child.action_type] = []
            action_branches[child.action_type].append(child)
        
        # Select the best trajectory from each action type
        for action_type, nodes in action_branches.items():
            if not nodes:
                continue
                
            # Sort by visits
            sorted_nodes = sorted(nodes, key=lambda n: n.visits, reverse=True)
            best_node = sorted_nodes[0]
            best_leaf = get_best_leaf(best_node)
            
            if best_leaf.visits > 0:  # Ensure the leaf node has been visited
                best_trajectories.append({
                    "action_type": action_type,
                    "action_name": self.action_templates[action_type]["name"],
                    "trajectory": best_leaf.get_formatted_trajectory(),
                    "visits": best_leaf.visits,
                    "value": best_leaf.get_value(),
                    "action_sequence": best_leaf.get_action_sequence()
                })
        
        # Sort by value, select the top top_n
        best_trajectories = sorted(best_trajectories, key=lambda t: t["value"], reverse=True)
        return best_trajectories[:top_n]
    
    def format_output(self, trajectories, root_query, search_tree):
        """Format the output result, fix the step display problem"""
        formatted_output = []
        
        # Add the search tree visualization
        formatted_output.append("## MCTS search tree visualization\n")
        formatted_output.append("```")
        formatted_output.append(search_tree)
        formatted_output.append("```\n")
        
        # Add the detailed information of each trajectory
        for i, traj in enumerate(trajectories):
            # Extract the action sequence
            action_sequence = traj["action_sequence"]
            action_sequence_str = " → ".join([a for a in action_sequence if a]) if action_sequence else "No action"
            
            output = f"## Candidate Trajectory {i+1}:\n"
            output += f"Action Sequence: {action_sequence_str}\n"
            output += f"Visits: {traj['visits']}, Value: {traj['value']:.2f}\n\n"
            
            # Add the detailed steps of the trajectory - this is the main fix point
            trajectory_full = traj["trajectory"]
            steps = []
            
            # Use regex or a more precise method to split the steps
            # First split out the query part and the steps part
            if "Query: " in trajectory_full:
                parts = trajectory_full.split("Query: ", 1)
                query_part = "Query: " + parts[1].split("\n\n", 1)[0]
                steps_part = parts[1].split("\n\n", 1)[1] if "\n\n" in parts[1] else ""
            else:
                query_part = trajectory_full
                steps_part = ""
            
            # Display the query
            query_text = query_part.replace('Query: ', '')
            output += f"**Original Query**: {query_text}\n\n"
            
            # Use a more precise method to split the steps content
            step_pattern = r"Step (\d+): \[Action ([A-Z0-9]+)\] (.*?)(?=Step \d+:|$)"
            import re
            step_matches = re.findall(step_pattern, steps_part, re.DOTALL)
            
            # Record the current step number
            current_step = 1
            
            # Process each step match
            for step_num, action_type, content in step_matches:
                
                action_name = self.action_templates[action_type]['name'] if action_type in self.action_templates else "Unknown Action"
                
                # Output the entire step content - ensure all text is included
                output += f"**Step {current_step} [{action_type}]**: {action_name}\n"
                output += f"{content.strip()}\n\n"
                current_step += 1
            
            # If the regex does not match any steps, try the traditional method
            if not step_matches and "\n\nStep " in trajectory_full:
                steps = trajectory_full.split("\n\nStep ")
                for step_idx, step in enumerate(steps[1:], 1):  # Skip the first element (query)
                    if "[Action " in step:
                        action_start = step.find("[Action ")
                        action_end = step.find("]", action_start)
                        
                        if action_start >= 0 and action_end >= 0:
                            action_type = step[action_start+8:action_end]
                            # Keep the entire step content, not just the first line
                            step_content = step[action_end+1:].strip()
                            
                            action_name = self.action_templates[action_type]['name'] if action_type in self.action_templates else "Unknown Action"
                            output += f"**Step {step_idx} [{action_type}]**: {action_name}\n"
                            output += f"{step_content}\n\n"
            
            # Add the final structured output
            # Ensure using the entire last step content, not the possibly truncated version
            last_step_content = ""
            if step_matches:
                last_step_content = step_matches[-1][2].strip()
            elif len(steps) > 1:
                last_step = steps[-1]
                if "[Action " in last_step:
                    action_end = last_step.find("]", last_step.find("[Action "))
                    if action_end >= 0:
                        last_step_content = last_step[action_end+1:].strip()
            
            # --- BEGIN MODIFICATION for format_output prompt ---
            # --- BEGIN MODIFICATION for format_output prompt ---
            format_prompt = f"""Synthesize the provided analysis to extract "entity name - numerical value - unit" triplets.
The goal is to capture any clearly named entity in the text that is directly quantified with BOTH a numerical value AND AN EXPLICIT UNIT.
**CRITICAL RULE: Only extract "name-value-unit" triplets where ALL THREE components (name, value, AND unit) are explicitly present and non-empty in the "final analysis" content. If an entity is mentioned with a numerical value but NO EXPLICIT UNIT is associated with it (e.g., 'pH 6', 'ratio 1.5'), that entity-value pair MUST BE DISCARDED and NOT extracted.**

The original query was: "{root_query}"
The final analysis from the reasoning process (this candidate trajectory) is:
---
{last_step_content}
---

**Key Extraction & Formatting Rules:**

1.  **Extraction Target (STRICT TRIPLET AND NON-EMPTY UNIT REQUIREMENT)**:
    * Identify any entity (material, parameter, etc.) that is explicitly named AND directly associated with BOTH a numerical value AND an EXPLICIT, NON-EMPTY unit in the provided "final analysis".
    * **If no explicit unit is associated with a numerical value for an entity, that entity-value pair is INVALID and MUST NOT be extracted.**
2.  **Entity Naming**:
    * Use the most descriptive name for the entity as suggested by the analysis.
    * The 'Material name' field MUST NOT contain the numerical value or unit.
    * **Chemical Formulas**: Extract only if the *entire compound itself* is given an explicit external quantity (value AND unit).
3.  **Quantities (MUST INCLUDE EXPLICIT UNIT)**:
    * Record the precise numerical value AND its corresponding EXPLICIT unit as found in the analysis.
    * The 'Quantity' field should ideally contain ONLY the numerical value [±Error if any]. The unit should ideally be in the 'Unit' field.
    * Include error margins if present with the numerical value.
4.  **Focus on Accuracy and Textual Grounding**:
    * Ensure all extracted triplets (name, value, unit) are directly supported by the "final analysis".
    * Avoid inventing information or misattributing quantities.
    * Do NOT treat citation numbers or unassociated identifiers as quantities.
5.  **Output Structure (FOLLOW THIS EXACTLY - Model should aim for this three-line structure per entry):**
    * Begin the entire output with "Material Information Summary:".
    * For each distinct **VALID (name, value, AND unit all present)** extracted triplet:
        - Material name: [Name of the Entity/Material/Parameter]
        - Quantity: [Numerical Value][±Error if any]
        - Unit: [Unit]
    * Use one empty line between different "Material Information Summary:" blocks.
    * NO markdown formatting.
6.  **Multiple Quantifications for One Entity**:
    * If a single entity has multiple distinct quantifications (each with a value AND unit), create SEPARATE entries for EACH valid (name-value-unit) triplet, repeating the entity name.

**Examples (ADHERE STRICTLY TO THIS THREE-LINE FORMAT PER ENTRY):**

(Example: analysis mentions "Ba (9.5 mmol) acetates")
Material Information Summary:
- Material name: Ba acetates
- Quantity: 9.5
- Unit: mmol

(Example: analysis mentions "drying was done at 120 degC for 2 h")
Material Information Summary:
- Material name: drying temperature
- Quantity: 120
- Unit: degC

Material Information Summary:
- Material name: drying time
- Quantity: 2
- Unit: h

(Example: analysis mentions "the pH was 6" -> **DO NOT EXTRACT (missing unit for '6')**)
(Example: analysis mentions "the ratio of A to B was 1.5" -> **DO NOT EXTRACT (missing unit for '1.5')**)
(Example: analysis mentions "The compound Ba0.95La0.05FeO3 was used." -> DO NOT EXTRACT if not externally quantified with value AND unit. If it says "5 grams of Ba0.95La0.05FeO3", then extract:
Material Information Summary:
- Material name: Ba0.95La0.05FeO3
- Quantity: 5
- Unit: grams

(Example: text "aqueous solution of RuCl3 (0.038 M, 5 mL)" - This is the format you requested for this specific example)
Material Information Summary:
- Material name: aqueous solution of RuCl3
- Quantity: 0.038
- Unit: M

Material Information Summary:
- Material name: aqueous solution of RuCl3
- Quantity: 5
- Unit: mL


Please extract all relevant "entity name - numerical value - explicit unit" triplets from the provided "final analysis" and structure them now according to all the rules above, especially aiming for the three-line output format per entry as shown in the examples. **Filter out any potential triplet that lacks an explicit unit for its numerical value, or is missing any of the three components.**
"""
            
            formatted_info = self.call_llm_api(format_prompt, max_tokens=2048)
            output += f"**Final Structured Output**:\n{formatted_info}\n"
            
            formatted_output.append(output)
            
        return "\n\n" + "\n\n".join(formatted_output)

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Material data extractor - LLM reasoning based on MCTS')
    parser.add_argument('--query', type=str, help='The material data text to be analyzed')
    parser.add_argument('--simulations', type=int, default=4, help='MCTS simulation times')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth') 
    parser.add_argument('--trajectories', type=int, default=3, help='Number of best trajectories to output')
    parser.add_argument('--temperature', type=float, default=0.7, help='Model temperature parameter')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Model maximum output token number')
    
    args = parser.parse_args()
    
    # Initialize the extractor, using the parameters passed in from the command line
    extractor = MaterialDataExtractor(max_tokens=args.max_tokens, temperature=args.temperature)
    
    # Print the action space definition
    extractor.print_action_space()
    
    # Get the query text
    query = args.query
    if not query:
        # If no query is provided in the command line, use the default query or prompt the user to input
        print("Please input the material data text to be analyzed (if empty, use the default example):")
        user_input = input().strip()
        if user_input:
            query = user_input
        else:
            query = """Extract triplets of material names, numerical values, and units from the text:
            "AgSbTe222 has a wide range of electronic properties of [Brag-BO]. A fully reproducible model-based inverse method (BO) revealed that the material has a bulk modulus (B0) of 156.3±5.1 GPa and pressure derivative (Bp) of 554.3±0.8 measured via ultrasonic techniques. Cu3Auadc has been estimated to have a bulk modulus of 1790 GPab via theoretical calculations."
            """
    
    # Run MCTS to generate trajectories
    print("Generating trajectories...")
    trajectories, root, search_tree = extractor.run_mcts(query, num_simulations=args.simulations, max_depth=args.depth)
    
    # Select the specified number of best trajectories
    if len(trajectories) > args.trajectories:
        trajectories = trajectories[:args.trajectories]
    
    # Format and print the result, including the search tree and action sequence
    formatted_output = extractor.format_output(trajectories, query, search_tree)
    print("\n\nGenerated Trajectories and Search Tree:")
    print(formatted_output)
    
    # Save the result
    with open("material_extraction_trajectories.txt", "w", encoding="utf-8") as f:
        f.write(formatted_output)
    print("\nResults saved to material_extraction_trajectories.txt")

if __name__ == "__main__":
    main()