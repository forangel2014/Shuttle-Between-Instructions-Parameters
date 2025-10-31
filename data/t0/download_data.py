from t0_config import DATA_SPLITS_SIZES
from datasets import load_dataset
import os
import json
import sys
import datasets
from multiprocessing import Pool
from functools import partial
config = datasets.DownloadConfig(resume_download=True, max_retries=100)
def process(key, out_dir):
    print(key)
    splits = DATA_SPLITS_SIZES[key].keys()
    print(splits)
    dataset = load_dataset('bigscience/P3', key, download_config=config)

    if not os.path.exists(os.path.join(out_dir, key)):
        os.makedirs(os.path.join(out_dir, key))

    for split in splits:
        d = dataset[split]
        input_ids, target_ids = [], []
        for dp in d:
            input_ids.append(dp["inputs_pretokenized"])
            target_ids.append(dp["targets_pretokenized"])

        out_file = os.path.join(out_dir, key, "{}_{}.json".format(key, split))
        with open(out_file, "w") as fout:
            json.dump([input_ids, target_ids], fout)

    
    # for split in splits:
    #     d = dataset[split]
    #     input_ids, target_ids = [], []
    #     for dp in d:
    #         input_ids.append(dp["inputs"])
    #         target_ids.append(dp["targets"])

    #     out_file = os.path.join(out_dir, key, "{}_{}.json".format(key, split))
    #     with open(out_file, "w") as fout:
    #         json.dump([input_ids, target_ids], fout)

    print("=" * 40)

def main():
    out_dir = "../datadict"
    os.makedirs(out_dir, exist_ok=True)

    keys = DATA_SPLITS_SIZES.keys()

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    i = 0
    l = len(keys)
    keys_list = []
    for key in keys:
        keys_list.append(key)
    # for j, key in enumerate(keys):
    while i < l:
        if i < start:
            i += 1
            continue
        
        try:
            print("Processing {}/{}".format(i, len(keys)))
            process(keys_list[i], out_dir)
        except:
            i -= 1
        
        if i == end:
            break
        i += 1
        
if __name__ == "__main__":
    main()

        