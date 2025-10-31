import os
import json

from datasets import load_dataset
import datasets
from t0_config import DATA_SPLITS_SIZES
config = datasets.DownloadConfig(resume_download=True, max_retries=100)


def expand_dataset_to_prompts(datasets):
    prompt_names = list(DATA_SPLITS_SIZES.keys())
    # select prompts corresponding the the selected datasets
    selected_prompts = filter(
        lambda x: any([x.startswith(item) for item in datasets]
                      ) and not x.endswith("score_eval"),
        prompt_names
    )
    selected_prompts = list(selected_prompts)
    return selected_prompts


def process(key, out_dir):
    data = load_dataset("bigscience/P3", key, download_config=config)
    eval_data = data["validation"]
    out_lines = []
    for i, dp in enumerate(eval_data):
        out_dict = {
            "id": i,
            "task": key,
            "input": dp["inputs_pretokenized"].strip(),
            "output": dp["targets_pretokenized"].strip(),
            "options": dp["answer_choices"] if "answer_choices" in dp else [],
            "input_tokenized": dp['inputs'],
            "output_tokenized": dp['targets'],
        }
        out_lines.append(out_dict)

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir, key)
    with open(os.path.join(out_dir, "{}.json".format(key)), "w") as fout:
        json.dump({"Instances": out_lines}, fout, indent=4)


def main():
    eval_datasets = [
        "super_glue_wsc.fixed",
        "winogrande_winogrande_xl",
        "super_glue_cb",
        "super_glue_rte",
        "anli",
        "super_glue_copa",
        "hellaswag",
        "super_glue_wic"
    ]

    out_dir = "../data_p3_eval"
    filtered_prompt_names = expand_dataset_to_prompts(eval_datasets)

    i = 0
    l = len(filtered_prompt_names)
    # for j, key in enumerate(keys):
    while i < l:
        # for i, prompt_name in enumerate(filtered_prompt_names):
        try:
            print("Processing {}/{}".format(i + 1, len(filtered_prompt_names)))
            print(filtered_prompt_names[i])
            process(filtered_prompt_names[i], out_dir)
        except:
            i -= 1
        i += 1


if __name__ == "__main__":
    main()
