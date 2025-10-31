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


# out_dir = "../data"
# for i, key in tqdm(enumerate(DATA_SPLITS_SIZES.keys())):
#     print("Processing {}/{}".format(i, len(DATA_SPLITS_SIZES.keys())))
#     print(key)
#     split = "validation"
#     data_file = os.path.join(out_dir, key, "{}_{}.json".format(key, split))
#     if not os.path.exists(data_file):
#         continue
#     with open(data_file, "r") as fin:
#         ds = []
#         input_ids, target_ids = json.load(fin)
#         for id, (i, t) in enumerate(zip(input_ids, target_ids)):
#             d = {}
#             d['id'] = id
#             d['input'] = i
#             d['output'] = t
#             ds.append(d)
#         with open(os.path.join("../data_dict", f"{key}_{split}.json"), "w") as fout:
#             json.dump({"Instances": ds}, fout, indent=4)
from promptsource.templates import DatasetTemplates

prompts = DatasetTemplates('amazon_polarity_flattering_or_not')

import pdb
pdb.set_trace()
