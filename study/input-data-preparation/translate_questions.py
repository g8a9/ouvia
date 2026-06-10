from google import genai
import os
import json
from dataclasses import dataclass, asdict
from absl import flags, app
import re
import time

import pandas as pd
import logging


assert (
    "GEMINI_API_KEY" in os.environ
), "Please set the GEMINI_API_KEY environment variable"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

flags.DEFINE_string("input_file", required=True, default=None, help="File with conversation starters")
flags.DEFINE_enum(
    "system",
    "gemini-2.5-pro",
    ["gemini-2.5-pro", "gemma-3-12b-it", "gemma-3-27b-it"],
    "Define the system to use for generation",
)
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling")
flags.DEFINE_float("top_p", 0.70, "Top P for sampling")
flags.DEFINE_float("top_k", 100, "Top K for sampling")
flags.DEFINE_integer("start_index", 0, "Index to start generation from.")
flags.DEFINE_integer("n_trials", 10, "Number of reattempts")
flags.DEFINE_integer("max_tokens", 32000, "Number of tokens to generate")

FLAGS = flags.FLAGS


def format_prompt(source_seg: str) -> str:
    with open(f"prompts/translate.txt", "r") as file:
        trans_prompt_template = file.read()

    trans_prompt = trans_prompt_template.format(source_seg=source_seg)

    return trans_prompt

def parse_qa_pairs(text):
  """Parses a string containing Q: and A: pairs into a list of (question, answer) tuples."""
  qa_pairs = []
  # Find all Q: and A: pairs
  matches = re.findall(r'Q: (.*?)\nA: (.*?)(?:\n|$)', text, re.DOTALL)
  for q, a in matches:
    qa_pairs.append((q.strip(), a.strip()))
  return qa_pairs


def main(args):
    
    df = pd.read_json(FLAGS.input_file, lines=True)

    config = genai.types.GenerateContentConfig(
        temperature=FLAGS.temperature,
        max_output_tokens=FLAGS.max_tokens,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        system_instruction="You are a language model specialized in translation generation.",
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
    output_file = f"outputs/trans_ques_{conv_type}_{FLAGS.system}.jsonl"
    
    all_responses = df['response'].to_list()
    
    with open(output_file, 'a+') as f:
        for i in range(FLAGS.start_index, len(all_responses)):
            for j, (q, _) in enumerate(parse_qa_pairs(all_responses[i])):
                prompt_curr = format_prompt(source_seg=q)
                response = None
                
                for trial in range(FLAGS.n_trials):
                    try:
                        response = client.models.generate_content(
                            model=FLAGS.system, contents=prompt_curr, config=config
                        )
                        logging.info(f"Trial {trial + 1}: {response}")

                        if response.candidates and response.candidates[0].finish_reason != "MAX_TOKENS":
                            # If we get a valid response, break the retry loop
                            break
                        else:
                            logging.warning(f"Trial {trial + 1} failed or hit MAX_TOKENS for prompt index {i}, qid {j}. Retrying...")
                            time.sleep(1) # Small delay before retrying
                    except Exception as e:
                        logging.error(f"An error occurred during trial {trial + 1} for prompt index {i}, qid {j}: {e}")
                        time.sleep(1) # Small delay before retrying
                # --- End of retry logic ---

                if response is None or not response.candidates:
                    logging.error(f"Failed to get a valid response after {FLAGS.n_trials} trials for prompt index {i}, qid {j}. Skipping.")
                    continue

                if response.candidates[0].finish_reason == "MAX_TOKENS":
                    logging.warning(f"Response hit MAX_TOKENS after {FLAGS.n_trials} trials for prompt index {i}, qid {j}. Skipping.")
                    continue

                data_dict = {
                    "index": i,
                    "qid": j,
                    "prompt": prompt_curr,
                    "response": response.text,
                }
                json.dump(data_dict, f)
                f.write('\n')
    print(f"Saved translated questions to {output_file}.")


if __name__ == "__main__":
    app.run(main)
