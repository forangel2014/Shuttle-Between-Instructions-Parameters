import copy
import openai
from openai import OpenAI
import os
import torch
import datasets
import re
import json
import numpy as np
import subprocess
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import string
from src.rouge import rouge_scorer

openai.api_key = os.getenv("OPENAI_API_KEY")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def hook(grad, name=None):
    if name:
        print(name)
    print(grad)

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    result = f"{hours}h{minutes}m{remaining_seconds}s"
    return result

def my_chat_template(messages, tokenize=None):
    text = ""
    for message in messages:
        text += f"{message['role']}: {message['content']}\n"
    return text

def get_gpu_memory_usage():
    # 使用nvidia-smi命令获取显存信息
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    # 解析输出
    output = result.stdout.decode('utf-8')
    lines = output.strip().split('\n')
    
    # 格式化输出
    info = ""

    for i, line in enumerate(lines):
        used, total = line.split(',')
        info += f"GPU {i}: {used} MiB / {total} MiB\n"

    return info

def post_process_for_prompting(predicted_knowledge):
    predicted_knowledge = predicted_knowledge.split("\n")[0]
    if len(predicted_knowledge.split(" ")[0]) == 1:
        predicted_knowledge = predicted_knowledge[2:]
    return predicted_knowledge

def post_process_for_y(y_pred):
    if not y_pred.startswith("<output>"):
        y_pred = "<output>" + y_pred
    if not y_pred.endswith("</output>"):
        y_pred = y_pred + "</output>"
    return y_pred

def plot_loss_curve(loss_dict, name):
    # 创建一个新的图形
    plt.figure()

    # 遍历损失字典中的每个损失
    for loss_name, loss_values in loss_dict.items():
        # 绘制损失曲线
        plt.plot(range(len(loss_values)), loss_values, label=loss_name)

    # 添加图例
    plt.legend()

    # 设置图形标题和轴标签
    plt.title('Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # 显示图形
    plt.savefig(f"{name}.pdf")

def tsne(encoded_latent, trained_latent, randomn_latent, filename):

    # Assume encoded_latent and trained_latent are lists of tensors
    # Convert the lists to numpy arrays
    encoded_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in encoded_latent]
    trained_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in trained_latent]
    randomn_latent = [tensor.to(torch.float16).detach().cpu().numpy() for tensor in randomn_latent]

    encoded_latent_np = np.array(encoded_latent)
    trained_latent_np = np.array(trained_latent)
    randomn_latent_np = np.array(randomn_latent)

    # Flatten the tensors to 2D arrays
    encoded_latent_flat = encoded_latent_np.reshape((len(encoded_latent), -1))
    trained_latent_flat = trained_latent_np.reshape((len(trained_latent), -1))
    randomn_latent_flat = randomn_latent_np.reshape((len(randomn_latent), -1))

    # Combine the flattened arrays
    combined_data = np.vstack((encoded_latent_flat, trained_latent_flat, randomn_latent_flat))

    # Apply t-SNE to reduce the dimensions to 2
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_result = tsne.fit_transform(combined_data)

    # Plotting the t-SNE visualization
    plt.figure(figsize=(8, 6))

    # Plot the encoded_latent points
    plt.scatter(tsne_result[:5, 0], tsne_result[:5, 1], label='encoded_latent')

    # Plot the trained_latent points
    plt.scatter(tsne_result[5:10, 0], tsne_result[5:10, 1], label='trained_latent')
    
    plt.scatter(tsne_result[10:, 0], tsne_result[10:, 1], label='randomn_latent', color="gray")

    # Add legend
    plt.legend()

    # Show the plot
    mkdir(os.path.dirname(filename))
    plt.savefig(filename)

def create_task_data_lookup(data):

    seen_train_knowledge_base = []
    
    for train_sample in data["seen_tasks"]["train"]:
        
        seen_train_knowledge_base.append(train_sample["knowledge"])
        
    knowledge = list(set(seen_train_knowledge_base))

    lookup = dict(zip(knowledge, list(range(len(knowledge)))))

    return knowledge, lookup

def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def remove_html(text):
        return re.sub(r'<[^>]*>', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_html(s))))

def match_p3(rule, input_):
    # 搜索rule中所有{{}}的模式
    pattern = r'{{.*?}}'
    matches = re.findall(pattern, rule)
    # 对于rule，将{{}}中的内容去掉
    idxs = [rule.find(match) for match in matches]
    rule_pieces = []
    for i in range(len(idxs)):
        if i == 0:
            rule_pieces.append(rule[:idxs[i]])
        else:
            rule_pieces.append(rule[idxs[i-1]+len(matches[i-1]):idxs[i]])
    rule_pieces.append(rule[idxs[-1]+len(matches[-1]):])
    # 丢弃少于两个单词的rule_pieces
    rule_pieces = [piece for piece in rule_pieces if len(piece.split()) > 1]
    rule = "".join(rule_pieces)
    for rule_piece in rule_pieces:
        input_ = input_.replace(rule_piece, "")
    return rule, input_

def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure

def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def load_task_data(task, unseen_task_ratio=None, unseen_task_num=None, test_sample_ratio=None, test_sample_num=None,
                   num_words=32, num_pertask=1000, task_fields=None):
    
    all_data = {
        "seen_tasks": {
            "train": [],
            "test": []
        },
        "unseen_tasks": {
            "train": [],
            "test": []
        },
        "prompt_template": None,
        "neural_evaluater": None,
        "symbolic_evaluater": None,
        "task_num": None,
        "seen_task_num": None
    }
    
    if task == "list_functions":
        
#         prompt_template = """please predict the output list given the input list.
# input: {}
# output: """

        prompt_template = "{}"
        all_data["prompt_template"] = prompt_template
        
        all_task_id = list(range(1, 251))
        task_num = len(all_task_id)
        random.shuffle(all_task_id)
        
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
            
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in range(1, 251):

            task_type = task_id2task_type[sub_task_id]
            
            task_dir = f"./data/{task}/c{sub_task_id:03d}"
            task_file = os.path.join(task_dir, "task.json")

            with open(task_file) as f:
                sub_task_data = json.load(f)

            description = sub_task_data["description"]
            examples = sub_task_data["examples"]

            pattern = r'"([^"]*)"'
            rule = re.findall(pattern, description)[0]
        
            sample_num = len(examples)
            all_sample_id = list(range(sample_num))
            random.shuffle(all_sample_id)
            
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")

            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
        
                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": example["input"],
                    "target": example["target"],
                    "knowledge": rule,
                    #"metadata": task_data,
                })
        
            if sub_task_id == 1:
                
                output_regex = sub_task_data["output_regex"]
                
                def neural_evaluater(y_pred, y_true, x, k):
                    matched = re.findall(output_regex, y_pred)
                    if len(matched):
                        return int(matched[0] == str(y_true))
                    else:
                        return 0

                def symbolic_evaluater(knowledge_pred, knowledge_true):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are two transformations described in natural language for lists. 
Please help me determine if these two transformations are equivalent.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                        except:
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score

                all_data["neural_evaluater"] = neural_evaluater
                all_data["symbolic_evaluater"] = symbolic_evaluater

    elif task == "sni":
        with open('src/data_dict.json', 'r') as f:
            data_dict = json.load(f)
            data_map = data_dict['data_map']
        
#         prompt_template = """Please complete the following task given the input and only return the output without other words.
# Input: {}
# Output: """

        prompt_template = "{}"
        all_data["prompt_template"] = prompt_template
        with open(f"./data/{task}/tasks_{num_words}.txt") as f:
            train_tasks = f.readlines()
            
        if task_fields is not None:
            task_fields = [data_map[t] for t in task_fields.split(',')]
            tasks = copy.deepcopy(train_tasks)
            train_tasks = []
            for task_name in tasks:
                task_file = f"./data/{task}/tasks/{task_name.strip()}.json"
                with open(task_file) as f:
                    sub_task_data = json.load(f)
                    if sub_task_data["Categories"][0] in task_fields:
                        train_tasks.append(task_name)
                
        all_task_id = list(range(1, len(train_tasks) + 1))
        task_num = len(all_task_id)
        print(f"task_num: {task_num}")
        random.shuffle(all_task_id)
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in range(1, len(train_tasks) + 1):

            task_type = task_id2task_type[sub_task_id]
            task_file = f"./data/{task}/tasks/{train_tasks[sub_task_id - 1].strip()}.json"
            with open(task_file) as f:
                sub_task_data = json.load(f)

            description = sub_task_data["Definition"][0]
            examples = []
            for ex in sub_task_data["Instances"]:
                if len(ex['input'].split(' ')) < 20:
                    examples.append(ex)
                # else:
                #     print(len(ex['input'].split(' ')))
                if len(examples) == num_pertask:
                    break
            if len(examples) != num_pertask:
                print(f"task_name: {train_tasks[sub_task_id - 1].strip()}, task_num: {len(examples)} is not enough")
                #if len(examples) < 60:
                continue
            # examples = sub_task_data["Instances"][:num_pertask]
            rule = description
            
            all_sample_id = list(range(len(examples)))
            sample_num = len(all_sample_id)
            random.shuffle(all_sample_id)
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")
            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
                input_ = example["input"] + "." if not example["input"][-1] in string.punctuation else example["input"]
                output = random.choice(example["output"])
                if output == '':
                    continue
                if not output[-1] in string.punctuation:
                    output += "."

                # add format to help LLM understand the task
                input_ = f"<input>{input_}</input>"
                output_ = f"<output>{output}</output>"
                rule_ = f"<instruction>{rule}</instruction>"

                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": input_,
                    "target": output_,
                    "knowledge": rule_,
                    #"metadata": task_data,
                })
        
            if sub_task_id == 1:
                
                def symbolic_evaluater(knowledge_pred, knowledge_true):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are two instructions described in natural language. 
Please help me determine if these two instructions are equivalent.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                        except Exception as e:
                            print(e)
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score
                
                # def neural_evaluater(y_pred, y_true):
                #     return (normalize_answer(y_pred.split('\n')[0]) == normalize_answer(y_true))

                def neural_evaluater(y_pred, y_true, x, k):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are an instruction, an input, an reference answer and a predicted answer.
Please help me determine if the predicted answer is correct.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
instruction: {k}
input: {x}
reference answer: {y_true}
predicted answer: {y_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                        except Exception as e:
                            print(e)
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score

                all_data["neural_evaluater"] = neural_evaluater
                all_data["symbolic_evaluater"] = symbolic_evaluater


    elif task == "instruction_induction":
        train_tasks = []
        for task in os.listdir('data/instruction_induction/annotations'):
            train_tasks.append(task.split('.json')[0])
        
#         prompt_template = """Please complete the following task given the input and only return the output without other words.
# Input: {}
# Output: """

        prompt_template = "{}"
        all_data["prompt_template"] = prompt_template
                
        all_task_id = list(range(1, len(train_tasks) + 1))
        task_num = len(all_task_id)
        print(f"task_num: {task_num}")
        random.shuffle(all_task_id)
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in range(1, len(train_tasks) + 1):

            task_type = task_id2task_type[sub_task_id]
            task_file = f"data/instruction_induction/raw/execute/{train_tasks[sub_task_id - 1]}.json"
            with open(task_file) as f:
                sub_task_data_execute = json.load(f)
            task_file = f"data/instruction_induction/raw/induce/{train_tasks[sub_task_id - 1]}.json"
            with open(task_file) as f:
                sub_task_data_induce = json.load(f)
            sub_task_data = list(sub_task_data_execute["examples"].values()) + list(sub_task_data_induce["examples"].values())

            knowledge_file = f"data/instruction_induction/annotations/{train_tasks[sub_task_id - 1]}.json"
            with open(knowledge_file) as f:
                knowledge_data = json.load(f)
            knowledge = knowledge_data['annotations'][0]
            examples = sub_task_data
            rule = knowledge

            all_sample_id = list(range(len(examples)))
            sample_num = len(all_sample_id)
            random.shuffle(all_sample_id)
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")
            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
                if "input" in example.keys():
                    input_ = example["input"] + "." if not example["input"][-1] in string.punctuation else example["input"]
                    output = example["output"]
                elif "cause" in example.keys():
                    input_ = example["cause"]
                    output = example["effect"]
                elif "concept" in example.keys():
                    input_ = ", ".join(example["items"])
                    output = example["concept"]
                else:
                    raise Exception("No input found in the example")
                if output == '':
                    continue
                if not output[-1] in string.punctuation:
                    output += "."

                # add format to help LLM understand the task
                input_ = f"<input>{input_}</input>"
                output_ = f"<output>{output}</output>"
                rule_ = f"<instruction>{rule}</instruction>"

                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": input_,
                    "target": output_,
                    "knowledge": rule_,
                    #"metadata": task_data,
                })
        
            if sub_task_id == 1:
                
                def symbolic_evaluater(knowledge_pred, knowledge_true):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are two instructions described in natural language. 
Please help me determine if these two instructions are equivalent.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                        except Exception as e:
                            print(e)
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score
                
                # def neural_evaluater(y_pred, y_true):
                #     return (normalize_answer(y_pred.split('\n')[0]) == normalize_answer(y_true))

                def neural_evaluater(y_pred, y_true, x, k):
                    messages = [
                        {
                        "role": "system",
                        "content": 
"""
Here are an instruction, an input, an reference answer and a predicted answer.
Please help me determine if the predicted answer is correct.
Only return \"True\" or \"False\".                        
"""
                        },
                        {
                        "role": "user",
                        "content": 
f"""
instruction: {k}
input: {x}
reference answer: {y_true}
predicted answer: {y_pred}
"""
                        },
                               ]
                    response = None
                    while not response:
                        try:
                            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                        except Exception as e:
                            print(e)
                            pass
                    response = response.choices[0].message.content
                    #print(response)
                    score = 1 if "true" in response.lower() else 0
                    return score

                all_data["neural_evaluater"] = neural_evaluater
                all_data["symbolic_evaluater"] = symbolic_evaluater



    elif task == "p3":
        """
        from src.t0_config import DATA_SPLITS_SIZES
        def load_dataset_names(task, split):
            with open(f'src/{task}.json', "r") as f:
                config = json.load(f)
            datasets = config[split]
            return datasets

        def expand_dataset_to_prompts(datasets):
            prompt_names = list(DATA_SPLITS_SIZES.keys())
            # select prompts corresponding the the selected datasets
            selected_prompts = filter(
                lambda x: any([x.startswith(item) for item in datasets]),
                prompt_names
            )
            selected_prompts = list(selected_prompts)
            return selected_prompts
        dataset_names = load_dataset_names("t0", "train")
        if task_fields is not None:
            dataset_names = task_fields.split(',')
        train_tasks = expand_dataset_to_prompts(dataset_names)

        # get rule from promptsource
        # from promptsource.templates import DatasetTemplates, TemplateCollection
        # collection = TemplateCollection()
        # prompts = collection.datasets_templates
        # res = {}
        # for task in train_tasks:
        #     for t in dataset_names:
        #         if task.startswith(t):
        #             name = task.split(f'{t}_')[1]
        #             if t == 'paws':
        #                 name = task.split(f'{t}_labeled_final_')[1]
        #             if name.endswith('_'):
        #                 name = name[:-1] + ' '
        #             flag = 0
        #             for prompt in prompts.keys():
        #                 # breakpoint()
        #                 if prompt[1] is not None:
        #                     p_name = prompt[0] + '_' + prompt[1]
        #                     pp_name = prompt[0] + '/' + prompt[1]
        #                 else:
        #                     p_name = prompt[0]
        #                     pp_name = prompt[0]
        #                 if 'art' == p_name:
        #                     continue
        #                 if 'quora' == p_name:
        #                     continue
        #                 if p_name in task:
        #                     flag = 1
                            # print(prompt)
        #                     if name == 'expand_reverse_task ':
        #                         name = "expand (reverse task)"
        #                     if name == 'Topic_Prediction_Answer_Only':
        #                         name = "Topic Prediction - Answer Only"
        #                     if name == 'Topic_Prediction_Question_Only':
        #                         name = "Topic Prediction - Question Only"
        #                     if name == 'Topic_Prediction_Question_and_Answer_Pair':
        #                         name = "Topic Prediction - Question and Answer Pair"
        #                     if name == 'Is_This_True ':
        #                         name = "Is This True?"
        #                     if name == 'Direct_Question_Closed_Book ':
        #                         name = "Direct Question (Closed Book)"
        #                     if name == 'Multiple_Choice_Closed_Book ':
        #                         name = "Multiple Choice (Closed Book)"
        #                     if name == 'PAWS_ANLI_GPT3':
        #                         name = "PAWS-ANLI GPT3"
        #                     if name == 'PAWS_ANLI_GPT3_no_label':
        #                         name = "PAWS-ANLI GPT3-no-label"
        #                     if name == 'task_description_no_label':
        #                         name = "task_description-no-label"
        #                     if name == 'Summarize ':
        #                         name = "Summarize:"
        #                     if name == 'Summarize_this_dialogue ':
        #                         name = "Summarize this dialogue:"
        #                     try:
        #                         rules = DatasetTemplates(pp_name)
        #                         rule = rules[name].jinja
        #                         res[task] = {'rule': rule, 'prompt': name, 'x': pp_name}
        #                     except:
        #                         try:
        #                             rules = DatasetTemplates(pp_name)
        #                             if task == 'common_gen_Given_concepts_type_2':
        #                                 name = 'Given concepts - type 2'
        #                             else:
        #                                 name = name.replace('_', ' ')
        #                             rule = rules[name].jinja
        #                             res[task] = {'rule': rule, 'prompt': name, 'x': pp_name}
        #                         except:
        #                             try:
        #                                 rules = DatasetTemplates(pp_name)
        #                                 name = name.replace(' ', '-')
        #                                 rule = rules[name].jinja
        #                                 res[task] = {'rule': rule, 'prompt': name, 'x': pp_name}
        #                             except:
        #                                 breakpoint()
        #             if not flag:
        #                 print("error   " + task + '  ' + t)
        #                 res[task] = 'none'
        # with open('src/t0_prompt.json', 'w') as f:
        #     json.dump(res, f, indent=4)
        with open('src/t0_prompt.json', 'r') as f:
            rules = json.load(f)
        # breakpoint()
        
        all_task_id = list(range(1, len(train_tasks) + 1))
        task_num = len(all_task_id)
        print(f"task_num: {task_num}")
        random.shuffle(all_task_id)
        if unseen_task_ratio:
            seen_task_num = max(round(task_num*(1-unseen_task_ratio)), 1)
        else:
            try:
                seen_task_num = task_num - unseen_task_num
            except:
                raise Exception("Neither unseen_task_ratio nor unseen_task_num is specified")
        task_id2task_type = dict([(id_, "seen_tasks") for id_ in all_task_id[:seen_task_num]] + [(id_, "unseen_tasks") for id_ in all_task_id[seen_task_num:]])
        
        for sub_task_id in tqdm(range(1, len(train_tasks) + 1)):

            rule = rules[train_tasks[sub_task_id - 1].strip()]['rule']
            rule = re.sub(r'{%.*?%}', '', rule)
            rule = rule.split("|||")[0]
            # 删除多余的换行符
            # print(rule)

            task_type = task_id2task_type[sub_task_id]
            task_file = f"../dataset/datadict/{train_tasks[sub_task_id - 1].strip()}/{train_tasks[sub_task_id - 1].strip()}_train.json"
            with open(task_file) as f:
                sub_task_data = json.load(f)
            sub_task_data = [{'input': sub_task_data[0][i], 'output': sub_task_data[1][i]} for i in range(len(sub_task_data[0]))]
            examples = []
            random.shuffle(sub_task_data)
            for ex in tqdm(sub_task_data):
                rule_new, input_ = match_p3(rule, ex['input'])
                output_ = ex['output'].strip("\n")

                if len(ex['input']) < 1000:
                    examples.append({'input': input_, 'output': output_})
                # else:
                #     print(len(ex['input'].split(' ')))
                if len(examples) == num_pertask:
                    break

            if len(examples) != num_pertask:
                print(f"task_name: {train_tasks[sub_task_id - 1].strip()}, task_num: {len(examples)} is not enough")
                #if len(examples) < 60:
                continue
            # examples = sub_task_data["Instances"][:num_pertask]

            all_sample_id = list(range(len(examples)))
            sample_num = len(all_sample_id)
            random.shuffle(all_sample_id)
            if test_sample_ratio:
                train_sample_num = round(sample_num*(1-test_sample_ratio))
            else:
                try:
                    train_sample_num = sample_num - test_sample_num
                except:
                    raise Exception("Neither test_sample_ratio nor test_sample_num is specified")
            sample_id2split = dict([(id_, "train") for id_ in all_sample_id[:train_sample_num]] + [(id_, "test") for id_ in all_sample_id[train_sample_num:]])
        
            for i in range(len(examples)):
                
                example = examples[i]
                split = sample_id2split[i]
                input_ = example["input"] + "." if not example["input"][-1] in string.punctuation else example["input"]
                output = example["output"]#random.choice(example["output"])
                if output == '':
                    continue
                if not output[-1] in string.punctuation:
                    output += "."

                # add format to help LLM understand the task
                input_ = f"<input>{input_}</input>"
                output_ = f"<output>{output}</output>"
                rule_ = f"<instruction>{rule_new}</instruction>"

                all_data[task_type][split].append({
                    "sub_task_id": sub_task_id,
                    "input": input_,
                    "target": output_,
                    "knowledge": rule_,
                    #"metadata": task_data,
                })
        """

        def symbolic_evaluater(knowledge_pred, knowledge_true):
            messages = [
                {
                "role": "system",
                "content": 
"""
Here are two instructions described in natural language. 
Please help me determine if these two instructions are equivalent.
Only return \"True\" or \"False\".                        
"""
                },
                {
                "role": "user",
                "content": 
f"""
transformation A: {knowledge_true}
transformation B: {knowledge_pred}
"""
                },
                        ]
            response = None
            while not response:
                try:
                    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                except Exception as e:
                    print(e)
                    pass
            response = response.choices[0].message.content
            #print(response)
            score = 1 if "true" in response.lower() else 0
            return score
        
        # def neural_evaluater(y_pred, y_true):
        #     return (normalize_answer(y_pred.split('\n')[0]) == normalize_answer(y_true))

        def neural_evaluater(y_pred, y_true, x, k):
            messages = [
                {
                "role": "system",
                "content": 
"""
Here are an instruction, an input, an reference answer and a predicted answer.
Please help me determine if the predicted answer is correct.
Only return \"True\" or \"False\".                        
"""
                },
                {
                "role": "user",
                "content": 
f"""
instruction: {k}
input: {x}
reference answer: {y_true}
predicted answer: {y_pred}
"""
                },
                        ]
            response = None
            while not response:
                try:
                    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
                except Exception as e:
                    print(e)
                    pass
            response = response.choices[0].message.content
            #print(response)
            score = 1 if "true" in response.lower() else 0
            return score

        all_data["seen_tasks"] = json.load(open(f"./data/{task}_seen_tasks.json", "r"))
        all_data["unseen_tasks"] = json.load(open(f"./data/{task}_unseen_tasks.json", "r"))
        # tasks that knowledge is not useful
        ban_task_ids = [1, 2, 4, 5, 6, 7, 9, 10, 11, 14, 15, 55, 56, 60, 61, 66, 67, 68, 71, 72, 73, 
        74, 86, 87, 89, 90, 96, 98, 99, 102, 103, 139, 140, 141, 142, 162, 165, 166, 167, 168, 182, 183, 
        184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 203, 
        204, 208, 209, 210, 212, 230, 232, 234, 240, 12, 93, 94, 211, 231, 256] 
        all_data["unseen_tasks"]["train"] = [task for task in all_data["unseen_tasks"]["train"] if task["sub_task_id"] not in ban_task_ids]
        all_data["unseen_tasks"]["test"] = [task for task in all_data["unseen_tasks"]["test"] if task["sub_task_id"] not in ban_task_ids]
        all_data["seen_tasks"]["train"] = [task for task in all_data["seen_tasks"]["train"] if task["sub_task_id"] not in ban_task_ids]
        all_data["seen_tasks"]["test"] = [task for task in all_data["seen_tasks"]["test"] if task["sub_task_id"] not in ban_task_ids]

        seen_task_num = len(all_data["seen_tasks"])
        task_num = len(all_data["seen_tasks"]) + len(all_data["unseen_tasks"])
        #json.dump(all_data["seen_tasks"], open(f"./data/{task}_seen_tasks.json", "w"), indent=4)
        #json.dump(all_data["unseen_tasks"], open(f"./data/{task}_unseen_tasks.json", "w"), indent=4)
        prompt_template = "{}"
        all_data["prompt_template"] = prompt_template
        all_data["neural_evaluater"] = neural_evaluater
        all_data["symbolic_evaluater"] = symbolic_evaluater

    else:
        raise Exception("Unknown dataset")

    # json.dump(all_data["seen_tasks"], open(f"./data/{task}_seen_tasks.json", "w"))
    # json.dump(all_data["unseen_tasks"], open(f"./data/{task}_unseen_tasks.json", "w"))
    all_data["seen_task_num"] = seen_task_num
    all_data["task_num"] = task_num
    print(f"seen_tasks: {seen_task_num}, unseen_tasks: {task_num - seen_task_num}")
    print(f"seen_tasks train number: {len(all_data['seen_tasks']['train'])}")
    print(all_data['seen_tasks']['train'][0])
    # import pdb; pdb.set_trace()
    return all_data

def load_pretrain_data_hf(pretrain_data_ratio, valid_ratio=0.1, valid_num=None, load_from_local=True, save=False):

    mkdir("./data/pretrain")

    all_samples = []

    # print("loading: Symbol-LLM/Symbolic_Collection")
    # path = "Symbol-LLM/Symbolic_Collection"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/Symbol-LLM-Symbolic_Collection.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(list(dataset["train"])[:10000]):
    #         my_sample = {
    #             "knowledge": sample["instruction"],
    #             "input": sample["input"],
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (Symbol-LLM-Symbolic_Collection): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/Symbol-LLM-Symbolic_Collection.json", "w"))


    print("loading: manythings-translations-alpaca")
    path = "xzuyn/manythings-translations-alpaca"
    if load_from_local:
        dataset_samples = json.load(open("./data/pretrain/manythings-translations-alpaca.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(list(dataset["train"])[:10000]):
            my_sample = {
                "knowledge": sample["instruction"],
                "input": sample["input"],
                "target": sample["output"],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (manythings-translations-alpaca): {len(dataset_samples)}")  
    all_samples.extend(dataset_samples)
    if save:
        json.dump(dataset_samples, open("./data/pretrain/manythings-translations-alpaca.json", "w"))

    print("loading: MBZUAI/LaMini-instruction")
    path = "MBZUAI/LaMini-instruction"
    if load_from_local:
        dataset_samples = json.load(open("./data/pretrain/LaMini-instruction.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(dataset["train"]):
            split_pos = sample["instruction"].lower().find("input:")
            input_pos = split_pos + len("input:")
            if split_pos == -1:
                split_pos = sample["instruction"].find(":")
                input_pos = split_pos + len(":")
            if split_pos == -1:
                continue
            knowledge, input_ = sample["instruction"][:split_pos], sample["instruction"][input_pos:]
            if len(input_) < 3:
                continue
            my_sample = {
                "knowledge": knowledge,
                "input": input_,
                "target": sample["response"],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (LaMini-instruction): {len(dataset_samples)}")  
    all_samples.extend(dataset_samples)
    if save:
        json.dump(dataset_samples, open("./data/pretrain/LaMini-instruction.json", "w"))

    print("loading: alpaca")
    path = "tatsu-lab/alpaca"
    if load_from_local:
        dataset_samples = json.load(open("./data/pretrain/alpaca.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(dataset["train"]):
            if sample["input"] == "":
                continue
            my_sample = {
                "knowledge": sample["instruction"],
                "input": sample["input"],
                "target": sample["output"],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (alpaca): {len(dataset_samples)}")  
    all_samples.extend(dataset_samples)
    if save:
        json.dump(dataset_samples, open("./data/pretrain/alpaca.json", "w"))

    print("loading: silk-road")
    path = "silk-road/alpaca-data-gpt4-chinese"
    if load_from_local:
        dataset_samples = json.load(open("./data/pretrain/silk-road.json"))
    else:
        dataset = datasets.load_dataset(path)
        dataset_samples = []
        for sample in tqdm(dataset["train"]):
            if sample["input"] == "":
                continue
            my_sample = {
                "knowledge": sample["instruction"],
                "input": sample["input"],
                "target": sample["output"],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (silk-road): {len(dataset_samples)}")  
    all_samples.extend(dataset_samples)
    if save:
        json.dump(dataset_samples, open("./data/pretrain/silk-road.json", "w"))

    print("loading: self-instruct")
    path = "yizhongw/self_instruct"
    if load_from_local:
        dataset_samples = json.load(open("./data/pretrain/self-instruct.json"))
    else:
        dataset = datasets.load_dataset(path, "super_natural_instructions")
        dataset_samples = []
        for sample in tqdm(list(dataset["train"]) + list(dataset["test"])):
            split_pos = sample["prompt"].find("Input:")
            if split_pos == -1:
                continue
            knowledge, input_ = sample["prompt"][:split_pos], sample["prompt"][split_pos+len("Input:"):]
            my_sample = {
                "knowledge": knowledge,
                "input": input_,
                "target": sample["completion"],
            }
            dataset_samples.append(my_sample)
    print(f"有效样本数量 (self-instruct): {len(dataset_samples)}")  
    all_samples.extend(dataset_samples) 
    if save:
        json.dump(dataset_samples, open("./data/pretrain/self-instruct.json", "w"))

    # print("loading: sail")
    # path = "sail/symbolic-instruction-tuning"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/sail.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(dataset["train"]):
    #         split_pos = sample["input"].find(":", len(sample["input"]) // 3 * 2)
    #         if split_pos == -1:
    #             split_pos = sample["input"].find("?", len(sample["input"]) // 3 * 2)
    #         if split_pos == -1:
    #             split_pos = sample["input"].find(".", len(sample["input"]) // 3 * 2)
    #         if split_pos == -1:
    #             split_pos = sample["input"].find("\n", len(sample["input"]) // 3 * 2)
    #         if split_pos == -1:
    #             continue
    #         knowledge, input_ = sample["input"][:split_pos+1], sample["input"][split_pos+1:]
    #         if len(input_) < 3 or len(knowledge) < 1 or len(sample["output"]) < 1:
    #             continue
    #         my_sample = {
    #             "knowledge": knowledge,
    #             "input": input_,
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (sail): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/sail.json", "w"))

    # path = "bigscience/xP3"
    # if load_from_local:
    #     dataset = datasets.load_from_disk("./data/pretrain/xP3")
    # else:
    #     dataset = datasets.load_dataset(path, "en")
    # dataset_samples = []
    # for sample in tqdm(dataset["train"]):
    #     print(sample)
    #     my_sample = {
    #         "knowledge": sample["instruction"],
    #         "input": sample["input"],
    #         "target": sample["output"],
    #     }
    #     dataset_samples.append(my_sample)
    # print(f"有效样本数量 (xP3): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     dataset.save_to_disk("./data/pretrain/xP3")

    # path = "BelleGroup/train_0.5M_CN"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/train_0.5M_CN.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(dataset["train"]):
    #         split_pos = sample["instruction"].find("。")
    #         if split_pos == -1:
    #             continue
    #         knowledge, input_ = sample["instruction"][:split_pos+1], sample["instruction"][split_pos+1:]
    #         if len(input_) < 3:
    #             continue
    #         my_sample = {
    #             "knowledge": knowledge,
    #             "input": input_,
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)        
    #     print(f"有效样本数量 (BelleGroup): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/train_0.5M_CN.json", "w"))

    # print("loading: orca")
    # path = "Open-Orca/OpenOrca"
    # if load_from_local:
    #     dataset = datasets.load_from_disk("./data/pretrain/orca")
    # else:
    #     dataset = datasets.load_dataset(path)
    # dataset_samples = []
    # for sample in tqdm(dataset["train"]):
    #     split_pos = sample["prompt"].find("\n", len(sample["prompt"]) // 3 * 2)
    #     if split_pos == -1:
    #         split_pos = sample["prompt"].find(".", len(sample["prompt"]) // 3 * 2)
    #     if split_pos == -1:
    #         split_pos = sample["prompt"].find(",", len(sample["prompt"]) // 3 * 2)
    #     if split_pos == -1:
    #         continue
    #     knowledge, input_ = sample["prompt"][:split_pos+1], sample["prompt"][split_pos+1:]
    #     if len(input_) < 3:
    #         continue
    #     my_sample = {
    #         "knowledge": knowledge,
    #         "input": input_,
    #         "target": sample["response"],
    #     }
    #     dataset_samples.append(my_sample)
    # print(f"有效样本数量 (orca): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     dataset.save_to_disk("./data/pretrain/orca")

    # print("loading: math")
    # path = "qwedsacf/grade-school-math-instructions"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/math.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(dataset["train"]):
    #         keywords = ["how", "what", "when"]
    #         knowledge = ""
    #         input_ = ""
    #         for keyword in keywords:
    #             match = re.search(r"\b" + keyword + r"\b", sample["INSTRUCTION"], re.IGNORECASE)
    #             if match:
    #                 start_pos = match.start()
    #                 knowledge = sample["INSTRUCTION"][:start_pos].strip()
    #                 input_ = sample["INSTRUCTION"][start_pos:].strip()
    #                 break
            
    #         if not knowledge or not input_:
    #             continue

    #         my_sample = {
    #             "knowledge": knowledge,
    #             "input": input_,
    #             "target": sample["RESPONSE"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (math): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/math.json", "w"))

    # print("loading: instruction")
    # path = "HuggingFaceH4/instruction-dataset"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/instruction.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(dataset["test"]):
    #         split_pos = sample["prompt"].find("\n", len(sample["prompt"]) // 3)
    #         if split_pos == -1:
    #             split_pos = sample["prompt"].find("?", len(sample["prompt"]) // 3)
    #         if split_pos == -1:
    #             split_pos = sample["prompt"].find(".", len(sample["prompt"]) // 3)
    #         if split_pos == -1:
    #             split_pos = sample["prompt"].find(",", len(sample["prompt"]) // 3)
    #         if split_pos == -1:
    #             continue
    #         knowledge, input_ = sample["prompt"][:split_pos+1], sample["prompt"][split_pos+1:]
    #         if len(input_) < 3:
    #             continue
    #         my_sample = {
    #             "knowledge": knowledge,
    #             "input": input_,
    #             "target": sample["completion"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (instruction): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/instruction.json", "w"))

    # print("loading: dolly")
    # path = "llm-wizard/dolly-15k-instruction-alpaca-format"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/dolly.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(dataset["train"]):
    #         if sample["input"] == "":
    #             continue
    #         my_sample = {
    #             "knowledge": sample["instruction"],
    #             "input": sample["input"],
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (dolly): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/dolly.json", "w"))

    # print("loading: planner")
    # path = "rewoo/planner_instruction_tuning_2k"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/planner.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(list(dataset["train"])[:100]):
    #         my_sample = {
    #             "knowledge": sample["instruction"],
    #             "input": sample["input"],
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (planner): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/planner.json", "w"))

    # print("loading: code")
    # path = "TokenBender/code_instructions_122k_alpaca_style"
    # if load_from_local:
    #     dataset_samples = json.load(open("./data/pretrain/code.json"))
    # else:
    #     dataset = datasets.load_dataset(path)
    #     dataset_samples = []
    #     for sample in tqdm(list(dataset["train"])[:10000]):
    #         if sample["input"] == "":
    #             continue
    #         my_sample = {
    #             "knowledge": sample["instruction"],
    #             "input": f"### Example Input:\n{sample['input']}",
    #             "target": sample["output"],
    #         }
    #         dataset_samples.append(my_sample)
    # print(f"有效样本数量 (code): {len(dataset_samples)}")  
    # all_samples.extend(dataset_samples)
    # if save:
    #     json.dump(dataset_samples, open("./data/pretrain/code.json", "w"))

    # elif dataset == "symbol":
    #     path = "Symbol-LLM/Symbolic_Collection"
    #     dataset = datasets.load_dataset(path)
    #     for sample in dataset["train"]:
    #         my_sample = {
    #             "knowledge": sample["instruction"],
    #             "input": sample["input"],
    #             "target": sample["output"],
    #         }
    #         all_samples.append(my_sample)

    all_samples = [sample for sample in all_samples if len(sample['input'].split(' ')) < 30 and len(sample['target'].split(' ')) < 30 and len(sample['knowledge'].split(' ')) < 30 and len(sample['knowledge'].split(' ')) > 1]

    for sample in all_samples:
        sample["input"] = f"<input>{sample['input']}</input>"
        sample["target"] = f"<output>{sample['target']}</output>"
        sample["knowledge"] = f"<instruction>{sample['knowledge']}</instruction>"

    random.shuffle(all_samples)

    all_samples = all_samples[:int(len(all_samples) * pretrain_data_ratio)]

    if valid_ratio:
        valid_num = max(round(len(all_samples)*valid_ratio), 1)
        
    print(f"train_num: {len(all_samples) - valid_num}, valid_num: {valid_num}")
    train_dataset = all_samples[:-valid_num]
    valid_dataset = all_samples[-valid_num:]
    return train_dataset, valid_dataset
