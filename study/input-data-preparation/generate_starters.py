import tyro
import pdb
from vllm import LLM, SamplingParams
import torch
import re
from dataclasses import dataclass, asdict

# For now, we return all candidates as if they are all valid
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
sentence_embedder = SentenceTransformer("all-mpnet-base-v2")


def load_conversations(scenario: str, n: int) -> str:
    with open("prompts/starters/base.txt", "r") as file:
        base_prompt = file.read()

    with open(f"prompts/starters/{scenario}.txt", "r") as file:
        scenario_prompt = file.read()

    user_prompt = base_prompt.format(scenario=scenario_prompt)

    conv = [
        {
            "role": "system",
            "content": "You are a language model specialized in generating high-quality synthetic data.",
        },
        {"role": "user", "content": user_prompt},
    ]

    return [conv] * n


@dataclass
class SamplingArgs:
    temperature: float = 0.6
    top_p: float = 0.95
    min_p: float = 0
    top_k: int = 20
    max_tokens: int = 8192


def main(
    scenario: str,
    model_name_or_path: str,
    total_starters: int = 100,
    micro_batch_size: int = 10,
    enable_thinking: bool = False,
    sampling_args: SamplingArgs = SamplingArgs(),
):

    # Iterative loop where we keep generate starters until "total_starters" starters valid are ready
    valid_starters = list()
    embedded_starters = None

    # Instatiate a vLLM model
    llm = LLM(
        model=model_name_or_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        tensor_parallel_size=4,
        enable_prefix_caching=True,  # if "gemma" not in model_name_or_path else False,
        # disable_sliding_window=True,
    )

    sampling_params = SamplingParams(**asdict(sampling_args))

    while len(valid_starters) < total_starters:

        # Generate micro_batch of starters
        convs = load_conversations(scenario=scenario, n=micro_batch_size)
        outputs = llm.chat(
            convs,
            sampling_params=sampling_params,
            chat_template_kwargs=(
                {"enable_thinking": enable_thinking} if enable_thinking else None
            ),
        )

        texts = [o.outputs[0].text.strip() for o in outputs]

        # If thinking is enabled, remove everything between <think> and </think>
        if enable_thinking:
            texts = [
                re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                for text in texts
            ]

        if embedded_starters is None:
            # If this is the first batch, we embed the starters
            embedded_starters = sentence_embedder.encode(texts)
        else:
            # If this is not the first batch, we filter the texts by similarity to the current pool
            embeddings = sentence_embedder.encode(texts)
            similarities = sentence_embedder.similarity(embedded_starters, embeddings)
            similarities = similarities.max(dim=0)[0]

            # Filter out texts that are too similar to existing starters
            threshold = 0.9  # Adjust this threshold as needed
            valid_indices = [i for i, sim in enumerate(similarities) if sim < threshold]

            if len(valid_indices) == 0:
                print(
                    "No valid starters found in this batch. Continuing to next batch."
                )
                continue
            else:
                print(f"Found {len(valid_indices)} valid starters in this batch.")

            texts = [texts[i] for i in valid_indices]
            embeddings = embeddings[valid_indices, :]

            # Update the embedded_starters with the new valid embeddings
            embedded_starters = np.vstack((embedded_starters, embeddings))

        valid_starters.extend(texts)
        # Check if the starter is valid
        # If valid, add to valid_starters

    print(f"Generated {len(valid_starters)} valid starters.")

    # Save the valid starters to a TSV file
    sanitized_model = model_name_or_path.replace("/", "--")
    output_file = f"outputs/{scenario}_{sanitized_model}_starters.tsv"
    df = pd.DataFrame({"starter": valid_starters})
    df.index.name = "id"
    df.to_csv(output_file, sep="\t")

    print(f"Saved {len(valid_starters)} starters to {output_file}.")


if __name__ == "__main__":
    tyro.cli(main)
