from google import genai
import os
from dataclasses import dataclass, asdict
from absl import flags, app

import pandas as pd
import logging


assert (
    "GEMINI_API_KEY" in os.environ
), "Please set the GEMINI_API_KEY environment variable"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

flags.DEFINE_enum(
    "system",
    "gemini-2.5-pro",
    ["gemini-2.5-pro", "gemma-3-12b-it", "gemma-3-27b-it"],
    "Define the system to use for generation",
)
flags.DEFINE_string("scenario", "pharmacy", "The scenario to use for phramacy")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling")
flags.DEFINE_float("top_p", 0.70, "Top P for sampling")
flags.DEFINE_float("top_k", 100, "Top K for sampling")
flags.DEFINE_integer("micro_batch_size", 5, "Number of outputs to generate")
flags.DEFINE_integer("max_tokens", 32000, "Number of tokens to generate")

FLAGS = flags.FLAGS


def load_user_prompt(scenario: str) -> str:
    with open("prompts/starters/base.txt", "r") as file:
        base_prompt = file.read()

    with open(f"prompts/starters/{scenario}.txt", "r") as file:
        scenario_prompt = file.read()

    user_prompt = base_prompt.format(scenario=scenario_prompt)

    return user_prompt


def main(args):
    user_prompt = load_user_prompt(scenario=FLAGS.scenario)

    config = genai.types.GenerateContentConfig(
        temperature=FLAGS.temperature,
        max_output_tokens=FLAGS.max_tokens,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        system_instruction="You are a language model specialized in generating high-quality synthetic data.",
        response_mime_type="text/plain",
        safety_settings=[
            genai.types.SafetySetting(
                category=category, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
            )
            for category in [
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ],
    )

    valid_starters = []
    count = 0
    while count != FLAGS.micro_batch_size:
        response = client.models.generate_content(
            model=FLAGS.system, contents=user_prompt, config=config
        )
        logging.info(f"{response}")
        if response.candidates is None:
            continue

        if response.candidates[0].finish_reason == "MAX_TOKENS":
            continue

        valid_starters.append(f"{response.text}")
        count += 1

    output_file = f"outputs/{FLAGS.scenario}_{FLAGS.system}_starters.tsv"
    df = pd.DataFrame({"starter": valid_starters})
    df.index.name = "id"
    df.to_csv(output_file, sep="\t")

    print(f"Saved {len(valid_starters)} starters to {output_file}.")


if __name__ == "__main__":
    app.run(main)
