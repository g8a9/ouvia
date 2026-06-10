import glob
import json

models_that_should_be_there = ["phi4mm", "voxtralsmall", "desta2", "tower-plus-9b"]
translation_files = glob.glob("./translations/*.jsonl")
# filenames are like "entry_<int-id>.jsonl". Sort them according to the int-id
translation_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

for tf in translation_files:
    with open(tf, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        models_in_file = set()
        for line in lines:
            data = json.loads(line)
            model_name = data.get("model", None)
            if model_name:
                models_in_file.add(model_name)

        missing_models = set(models_that_should_be_there) - models_in_file
        if missing_models:
            print(f"In file {tf}, the following models are missing translations: {missing_models}")