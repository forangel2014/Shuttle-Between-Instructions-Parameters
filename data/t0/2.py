import re
from _ctypes import PyObj_FromPtr
from datasets import load_dataset
from tqdm import tqdm
from t0_config import DATA_SPLITS_SIZES, eval
from datasets import load_dataset
import os
import json
import sys
import datasets
from multiprocessing import Pool
from functools import partial
config = datasets.DownloadConfig(resume_download=True, max_retries=100)


class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(
                no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


def expand_dataset_to_prompts(datasets):
    prompt_names = eval#list(DATA_SPLITS_SIZES.keys())
    # select prompts corresponding the the selected datasets
    selected_prompts = filter(
        lambda x: any([x.startswith(item) for item in datasets]
                      ) and not x.endswith("score_eval"),
        prompt_names
    )
    selected_prompts = list(selected_prompts)
    return selected_prompts


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
eval_datasets = ["story_cloze"]
out_dir = "../data_eval"
filtered_prompt_names = expand_dataset_to_prompts(eval_datasets)

i = 0
l = len(filtered_prompt_names)
# for j, key in enumerate(keys)
for i, key in tqdm(enumerate(filtered_prompt_names)):
    print("Processing {}/{}".format(i, l))
    print(key)
    split = "valid"
    data_file = os.path.join(out_dir, key, "{}_{}.jsonl".format(key, split))
    # if not os.path.exists(data_file):
    #     continue
    with open(data_file, "r") as fin:
        ds = []
        for id, line in enumerate(fin):
            data = json.loads(line)
            data['id'] = id
            data['input_tokenized'] = []
            data['output_tokenized'] = []
            ds.append(data)
    with open(os.path.join("../data_p3_eval_sto", f"{key}.json"), "w") as fout:
        json.dump({"Instances": ds}, fout, indent=4)
