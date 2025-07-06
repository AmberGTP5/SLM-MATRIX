import os
import json
import time
import requests
import openai
import copy

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")

MAX_TOKENS = os.getenv("MAX_TOKENS")
TEMPERATURE = os.getenv("TEMPERATURE")

DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    api_base=API_BASE,
    api_key=API_KEY,
    streaming=False,
):

    logger.info(
        f"Input data: model={model}, messages={messages}, max_tokens={max_tokens}, temperature={temperature}"
    )

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = f"{api_base}/chat/completions"

            logger.info(f"Sending request to {endpoint}")

            # Assuming model is a list with one element, e.g., ['qwen2']
            chat_model = model[0] if isinstance(model, list) else model

            res = requests.post(
                endpoint,
                json={
                    "model": chat_model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            logger.info(f"Response: {res.json()}")

            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    logger.info(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    api_base=API_BASE,
    api_key=API_KEY
):
    endpoint = api_base
    client = openai.OpenAI(api_key=api_key, base_url=endpoint)
    
    # Unified model parameters
    model_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature > 1e-4 else 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    response = client.chat.completions.create(**model_params)
    return response


def generate_openai(
    model,
    messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
):

    client = openai.OpenAI(
        base_url=API_BASE_2,
        api_key=API_KEY_2,
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    # system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.Responses from models:"""
    system = f"""Synthesize provided model responses to extract materials data and relevant process parameters.
Primary Task: Extract explicitly named entities (chemical substances, solutions, suspensions, gases, process parameters, key physical dimensions, etc.) that are directly and explicitly quantified with a numerical value AND an associated unit. Only "name-value-unit" triplets where ALL THREE components (name, value, unit) are non-empty and explicitly stated are valid. Follow all rules and the exact output format below.

Key Extraction & Formatting Rules:

Extraction Target (STRICT TRIPLET AND NON-EMPTY REQUIREMENT):
Identify any entity (material, parameter, key physical dimension, etc.) that is explicitly named AND directly associated with BOTH a numerical value AND a clearly stated unit in the provided text.
CRITICAL FILTER: If any part of the potential "name-value-unit" triplet (i.e., the entity name, the numerical value, or the unit) is missing, empty, or not explicitly found in the text, then that entire entry MUST BE DISCARDED and NOT included in the output.
If an entity is quantified with a numerical value but NO explicit unit is present (e.g., 'pH 6', 'ratio 1.5'), DO NOT extract it. The presence of a non-empty unit is mandatory.
This includes specific chemical names (e.g., 'oxalic acid (90 mmol)'), generic terms if quantified with value and unit (e.g., 'an aqueous solution (50 mL)'), quantified process parameters with value and unit (e.g., 'sintering temperature 1100 degC', 'drying time 2 h'), and relevant quantified physical dimensions with value and unit (e.g., 'film thickness 0.5 mm').
Quantified components within a described mixture/solution: Extract the parent mixture/solution if it's quantified (with name, value, and unit), AND ALSO extract its individual quantified components (with their own names, values, and units) as separate entries, ensuring each component also forms a complete triplet.
Material/Entity Naming (CRITICAL):
Use the full, specific name for the material or entity from the text. If a clear, explicit name for the entity directly associated with the value and unit cannot be confidently identified from the text, the triplet is invalid and MUST BE DISCARDED.
The 'Material name' (or 'Entity name') field MUST NOT contain the numerical value or unit that represents its quantity/value. This information belongs strictly in the 'Quantity' field.
Purity or concentration (e.g., '99% hydrazine monohydrate') can be part of the material name ONLY if it's clearly part of its textual identification before its main quantity (value and unit) is stated, and not the quantity itself.
Quantities (CRITICAL - MUST INCLUDE NON-EMPTY VALUE AND UNIT):
Record the precise numerical value AND its corresponding unit exactly as found in the text.
If no explicit numerical value or no explicit unit is associated with an identifiable entity name in the text, that potential entry is invalid and should NOT be extracted.
MUST include error margins (e.g., ±0.1) if they are present with the value in the text.
EXCLUDE (Do Not Extract - GUIDANCE ADJUSTED FOR STRICT TRIPLET AND NON-EMPTY):
Unquantified materials/entities.
Materials/entities mentioned with a numerical value but without an explicit, accompanying unit (e.g., pH values, ratios, counts without units).
Materials/entities mentioned with a unit but without an explicit, accompanying numerical value (e.g., "pressure in GPa").
Entries where a clear entity name cannot be determined alongside an explicit value and unit.
Vague or purely qualitative descriptions.
Citation numbers or figure/table numbers.
Subscript numbers within a chemical formula unless the entire compound itself is given an explicit external quantity (value AND unit).
Output Structure (FOLLOW THIS EXACTLY - User will post-process unit line if necessary):
Begin the entire output with "Material Information Summary:".
For each distinct valid and complete "name-value-unit" triplet (where name, value, AND unit are ALL non-empty and explicit):
Material name: [Name of the Entity/Material/Parameter]
Quantity: [Numerical Value][±Error if any] [Unit]
Unit: [Unit] // Model may still produce this; ensure Quantity line above contains the unit too.
Use one empty line between different "Material Information Summary:" blocks.
NO markdown formatting (like bold ** or italics *) in the "Material Information Summary:" section. Use a dash (-) for list items as shown.
Multiple Quantifications for One Material/Entity Instance:
If a single mention of a material/entity in the text has multiple direct forms of quantification (each forming a complete "name-value-unit" triplet), you MUST create SEPARATE "Material Information Summary:" entries for EACH valid triplet, repeating the material/entity name.
Examples (ADHERE STRICTLY TO THIS FORMAT AND EXTRACTION RULES):

(Example: text "Ba (9.5 mmol) acetates")
Material Information Summary:

Material name: Ba acetates
Quantity: 9.5
Unit: mmol
(Example: text "water (100 mL)")
Material Information Summary:

Material name: water
Quantity: 100
Unit: mL
(Example: text "sintered at 1100 degC for 5 h")
Material Information Summary:

Material name: sintering temperature
Quantity: 1100
Unit: degC
Material Information Summary:

Material name: sintering time
Quantity: 5
Unit: h
(Example: text "The thickness was about 0.5 mm")
Material Information Summary:

Material name: thickness
Quantity: 0.5
Unit: mm
(Example: text "aqueous solution of RuCl3 (0.038 M, 5 mL)")
Material Information Summary:

Material name: aqueous solution of RuCl3
Quantity: 0.038
Unit: M
Material Information Summary:

Material name: aqueous solution of RuCl3
Quantity: 5
Unit: mL
(Example: text "the pH was 6" -> DO NOT EXTRACT (missing unit for '6'))
(Example: text "ratio was 1.5" -> DO NOT EXTRACT (missing unit for '1.5'))
(Example: text "a BLF powder was used" -> DO NOT EXTRACT (missing value and unit for BLF powder's quantity/property))
(Example: text "pressure was high (GPa)" -> DO NOT EXTRACT (missing numerical value for pressure even if GPa is mentioned as a general unit context))

Final Instruction: Critically evaluate the information from the source model responses. Your final output must be a refined and accurate synthesis that strictly adheres to ALL instructions and rules provided above, especially the "name-value-unit" triplet requirement for extraction. Only output "Material Information Summary:" blocks for triplets where the Material name, the Numerical Value, AND the Unit are ALL explicitly present, identifiable, and non-empty in the source text. If any of these three components are missing or cannot be confidently determined for a potential extraction, that entire entry must be filtered out and entirely excluded from the final output.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"{reference}"

    messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    generate_fn=generate_together,
    api_base=API_BASE,
    api_key=API_KEY
):

    # Check if messages contain images
    has_image = any(isinstance(m.get('content'), list) and 
                    any(c.get('type') == 'image_url' for c in m.get('content', []))
                    for m in messages)
    
    if has_image:
        # If contains images, use messages directly without adding references
        full_messages = messages
        
    else:
        # If text only, add references (original logic)
        full_messages = inject_references_to_messages(messages, references)
    
    return generate_fn(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key
    )