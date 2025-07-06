from PIL import Image
import io
import datetime
import base64
import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
import typer
import os
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from datasets.utils.logging import disable_progress_bar
from time import sleep

from dotenv import load_dotenv

# load_dotenv()
def get_env_var(key):
    load_dotenv(override=True)
    return os.getenv(key)

API_KEY = get_env_var("API_KEY")
API_BASE = get_env_var("API_BASE")

API_KEY_2 = get_env_var("API_KEY_2")
API_BASE_2 = get_env_var("API_BASE_2")

MAX_TOKENS = get_env_var("MAX_TOKENS")
TEMPERATURE = get_env_var("TEMPERATURE")
ROUNDS = get_env_var("ROUNDS")
MULTITURN = get_env_var("MULTITURN") == "True"

MODEL_AGGREGATE = get_env_var("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = get_env_var("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = get_env_var("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = get_env_var("MODEL_REFERENCE_3")

# Create output directory if it doesn't exist
OUTPUT_DIR = "moa_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

disable_progress_bar()

# Initialize console with recording enabled
console = Console(record=True)

welcome_message = (
    """
## MoA (Mixture-of-Agents)

The following LLMs as reference models, then passes the results to the aggregate model for the final response:
- """
    + MODEL_AGGREGATE
    + """   <--- Aggregate model
- """
    + MODEL_REFERENCE_1
    + """   <--- Reference model 1
- """
    + MODEL_REFERENCE_2
    + """   <--- Reference model 2
- """
    + MODEL_REFERENCE_3
    + """   <--- Reference model 3

"""
)

default_reference_models = [
    # MODEL_AGGREGATE,
    MODEL_REFERENCE_1,
    MODEL_REFERENCE_2,
    MODEL_REFERENCE_3,
]


def process_fn(
    item,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
):

    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    # Use console.print instead of print
    console.print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except IOError:
        return None

def main(
    model: str = MODEL_AGGREGATE,
    reference_models: list[str] = default_reference_models,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    rounds: int = ROUNDS,
    multi_turn=MULTITURN,
):
    md = Markdown(welcome_message)
    console.print(md)
    sleep(0.75)
    console.print(
        "\n[bold]To use this demo, answer the questions below to get started [cyan](press enter to use the defaults)[/cyan][/bold]:"
    )

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    num_proc = len(reference_models)

    model = Prompt.ask(
        "\n1. What main model do you want to use?",
        default=MODEL_AGGREGATE,
    )
    console.print(f"Selected {model}.", style="yellow italic")
    temperature = float(
        Prompt.ask(
            "2. What temperature do you want to use?",
            default=TEMPERATURE,
            show_default=True,
        )
    )
    console.print(f"Selected {temperature}.", style="yellow italic")
    max_tokens = int(
        Prompt.ask(
            "3. What max tokens do you want to use?",
            default=MAX_TOKENS,
            show_default=True,
        )
    )
    console.print(f"Selected {max_tokens}.", style="yellow italic")

    current_input_type = "Text"  # Default to text input

    while True:
        current_input_type = Prompt.ask(
            "\n[cyan bold]Select input type[/cyan bold]",
            choices=["Text", "Image", "Exit"],
            default="Text"
        )

        if current_input_type == "Exit":
            # Use console.print instead of print
            console.print("Goodbye!")
            break

        while True:
            if current_input_type == "Text":
                while True:
                    instruction = Prompt.ask("\n[cyan bold]Text prompt [Enter EXIT T to quit] >>[/cyan bold] ")
                    if instruction.strip().lower() == "exit t":
                        break
                    if not instruction.strip():
                        console.print("[red]Input cannot be empty, please try again.[/red]")
                        continue
                    content = instruction
                    use_reference_models = True

                    # Logic for processing text input
                    if use_reference_models:
                        if multi_turn:
                            for i in range(len(reference_models)):
                                data["instruction"][i].append({"role": "user", "content": content})
                                data["references"] = [""] * len(reference_models)
                        else:
                            data = {
                                "instruction": [{"role": "user", "content": content}] * len(reference_models),
                                "references": [""] * len(reference_models),
                                "model": [m for m in reference_models],
                            }

                        eval_set = datasets.Dataset.from_dict(data)

                        with console.status("[bold green]Querying all models...") as status:
                            for i_round in range(rounds):
                                eval_set = eval_set.map(
                                    partial(
                                        process_fn,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                    ),
                                    batched=False,
                                    num_proc=num_proc,
                                )
                                references = [item["output"] for item in eval_set]
                                data["references"] = references
                                eval_set = datasets.Dataset.from_dict(data)

                    # Code to generate and output results
                    generate_and_output_result(model, temperature, max_tokens, content, references)

            elif current_input_type == "Image":
                while True:
                    image_path = Prompt.ask("\n[cyan bold]Image path [Enter EXIT P to quit] >>[/cyan bold] ")
                    if image_path.strip().lower() == "exit p":
                        break
                    if not image_path.strip():
                        console.print("[red]Input cannot be empty, please try again.[/red]")
                        continue
                    base64_image = encode_image(image_path)
                    if base64_image is None:
                        console.print("[red]Cannot recognize image path, please try again.[/red]")
                        continue

                    instruction = Prompt.ask("\n[cyan bold]Image description (optional) >>[/cyan bold] ")
                    content = [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                    use_reference_models = False # Typically reference models might not support images well

                    # Logic for processing image input
                    references = [] # No references for image input in this setup

                    # Code to generate and output results
                    generate_and_output_result(model, temperature, max_tokens, content, references)

            break  # Exit the inner loop for the current input type

def generate_and_output_result(model, temperature, max_tokens, content, references):
    console.print("[cyan bold]Aggregating results and querying the aggregated model...[/cyan bold]")
    output = generate_with_references(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
        references=references,
        generate_fn=generate_together_stream,
        api_base=API_BASE_2,
        api_key=API_KEY_2
    )

    # Console recording handles capturing the output stream
    console.print("\n")
    console.log(Markdown(f"## Final answer from {model}"))

    for chunk in output:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            out = chunk.choices[0].delta.content
            console.print(out, end="")
        elif hasattr(chunk, 'usage'):
            # This is the last chunk, containing usage information
            if DEBUG:
                logger.info(f"Final usage information: {chunk.usage}")
        # No need for the else branch as we've handled all expected cases

    console.print() # Print a final newline

    # Extract query for filename
    query_text = ""
    if isinstance(content, str):
        # For text queries
        query_text = content[:30].replace(" ", "_").replace("/", "_").replace("\\", "_")
    elif isinstance(content, list):
        # For image queries with optional text
        for item in content:
            if item.get("type") == "text" and item.get("text"):
                query_text = item["text"][:30].replace(" ", "_").replace("/", "_").replace("\\", "_")
                break
        if not query_text:
            query_text = "image_query"

    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if query_text:
        filename = f"{timestamp}_{query_text}.md"
    else:
        filename = f"{timestamp}_moa_result.md"

    # Clean filename of any problematic characters
    filename = "".join(c for c in filename if c.isalnum() or c in "_-.")
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Export the entire console output recorded so far
    # Use clear=False to keep the recording buffer if needed later,
    # True might be slightly cleaner if saving per run is the only goal.
    full_console_output = console.export_text(clear=False)

    # Save the exported console output to the file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_console_output)
        console.print(f"\n[green]Full console output saved to:[/green] {filepath}")
    except Exception as e:
        console.print(f"\n[red]Error saving file:[/red] {e}")


    if DEBUG:
        # Log the beginning of the captured output for debugging if needed
        logger.info(f"model: {model}, instruction: {content}, saved output starts with: {full_console_output[:100]}")

if __name__ == "__main__":
    typer.run(main)