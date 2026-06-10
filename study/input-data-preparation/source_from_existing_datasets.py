# %%
import json
from google import genai
import os
from tqdm import tqdm
import glob
import pandas as pd

# MED-MT: https://www.nature.com/articles/s41597-022-01423-1
data = []
for fname in glob.glob("../data/Clean Transcripts/*"):
    txt_name = fname.split("/")[-1]
    try:
        with open(fname, "r", encoding="utf-8") as f:
            patient_text = f.read().split("\n\n")[1]
        data.append([txt_name, len(patient_text.split(' ')), patient_text])
        # print(f"{txt_name}\t{len(patient_text.split(' '))}\t{patient_text}")
    except:
        continue

# %%
df = pd.DataFrame(data)
df.columns = ["fname", "len", "starter"]
sub_df = df[df.len >= 50]

# %%
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
config = genai.types.GenerateContentConfig(
    temperature=0.9,
    max_output_tokens=32768,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
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

# %%
responses = {}
for i, row in tqdm(sub_df.iterrows()):
    user_prompt = f"Please convert this transcribed text into a fluent text. Preserve as much medical information as possible. The resulting text should contain atleast 50-60 words.\n\n{str(row['starter'])}"
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=user_prompt, config=config
    )
    responses[i] = response.text

for key in responses.keys():
    print(f"{key}\t{len(responses[key])}\t{responses[key]}")


# %%
# TaskMaster En-De: https://github.com/Unbabel/BConTrasT/blob/master/test1_corpus-with_gold_references.json
with open("/content/BConTrasT/test1_corpus-with_gold_references.json") as f:
    dialogues = json.load(f)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
config = genai.types.GenerateContentConfig(
    temperature=0.9,
    max_output_tokens=32768,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
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

# %%
responses = {}
for key, dialogue in tqdm(dialogues.items()):
    user_prompt = f"Please convert this dialogue into a single turn request by the customer using all the details provided. The resulting text should be fluent text with atleast 50-60 words.\n\n{str(dialogue)}"
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=user_prompt, config=config
    )
    responses[key] = response.text

for key in responses.keys():
    print(f"{key}\t{len(responses[key])}\t{responses[key]}")
