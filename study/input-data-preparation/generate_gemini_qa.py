from google import genai
import os
import json
from dataclasses import dataclass, asdict
from absl import flags, app

import pandas as pd
import logging


assert (
    "GEMINI_API_KEY" in os.environ
), "Please set the GEMINI_API_KEY environment variable"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

flags.DEFINE_string("input_file", required=True, default=None, help="File with conversation starters")
flags.DEFINE_string("prompt_col", "starter", "Column corresponding to prompt text")
flags.DEFINE_enum(
    "system",
    "gemini-2.5-pro",
    ["gemini-2.5-pro", "gemma-3-12b-it", "gemma-3-27b-it"],
    "Define the system to use for generation",
)
flags.DEFINE_string("prompt_name", "treqa", "Prompt template for QAG")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling")
flags.DEFINE_float("top_p", 0.70, "Top P for sampling")
flags.DEFINE_float("top_k", 100, "Top K for sampling")
flags.DEFINE_integer("max_tokens", 32000, "Number of tokens to generate")

FLAGS = flags.FLAGS


def format_qag_prompt(prompt_name: str, passage: str) -> str:
    with open(f"prompts/qag/{prompt_name}.txt", "r") as file:
        qa_prompt_template = file.read()

    qa_prompt = qa_prompt_template.format(passage=passage)

    return qa_prompt


def main(args):
    
    df = pd.read_csv(FLAGS.input_file, sep="\t")
    qag_prompts = [format_qag_prompt(prompt_name=FLAGS.prompt_name, passage=x) for x in df[FLAGS.prompt_col] ]
    print(qag_prompts[0])

    config = genai.types.GenerateContentConfig(
        temperature=FLAGS.temperature,
        max_output_tokens=FLAGS.max_tokens,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        system_instruction="You are a language model specialized in generating high-quality question and answers.",
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

    conv_type = FLAGS.input_file.split("/")[-1].split(".")[0]
    output_file = f"outputs/qag_{FLAGS.prompt_name}_{conv_type}_qag_{FLAGS.system}.jsonl"
    
    with open(output_file, 'w') as f:
        for i in range(0, len(qag_prompts)):
            prompt_curr = qag_prompts[i]
            response = client.models.generate_content(
                model=FLAGS.system, contents=prompt_curr, config=config
            )
            logging.info(f"{response}")
            if response.candidates is None:
                continue

            if response.candidates[0].finish_reason == "MAX_TOKENS":
                continue
            data_dict = {
                "index": i,
                "prompt": prompt_curr,
                "response": response.text,
            }
            json.dump(data_dict, f)
            f.write('\n')
    print(f"Saved question-answers to {output_file}.")


if __name__ == "__main__":
    app.run(main)
