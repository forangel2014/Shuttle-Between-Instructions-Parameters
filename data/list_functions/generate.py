
import re
import os
import json
import openai

prompt = [
    {"role": "system",
     "content": 
"""
There is a transformation that transform an input list to an output list.
The transform has been described in natural language.
Please write a python function to implement it.
"""
    },
    {"role": "user",
     "content": "remove all but element 3"},
    {"role": "assistant",
     "content": 
"""
def transform(input_list):
    output_list = [input_list[2]]
    return output_list
"""
    },
]

def openai_api(model_id, messages):
    response = openai.chat.completions.create(
        model=model_id,
        messages=messages,
        # temperature=temperature,
        # max_tokens=self.max_token,
        # top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        #stop=stop
    )
    response = response.dict()
    return response["choices"][0]["message"]["content"]

def geenrate_instances(rule, example):
    
    messages = prompt + [{"role": "user", "content": rules}]

    translated_function = openai_api(model_id, messages)

    # Parse and evaluate the translated function
    exec(translated_function)
    transform_func = locals()["transform"]

    # Apply the function to each input and check against target
    is_valid = True
    for example in examples:
        input_list = eval(example["input"])
        target_list = eval(example["target"])
        output_list = transform_func(input_list)
        if output_list != target_list:
            is_valid = False
            break

    if is_valid:
        # Save the function
        save_path = os.path.join(task_dir, "function.py")
        with open(save_path, "w") as f:
            f.write(translated_function)
    else:
        print(f"Function for {task_dir} is not valid. Please modify it.")

def process_task_json(task_id):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = f"{file_dir}/c{task_id:03d}"
    task_file = os.path.join(task_dir, "task.json")

    with open(task_file) as f:
        task_data = json.load(f)

    description = task_data["description"]
    examples = task_data["examples"]

    pattern = r'"([^"]*)"'
    rule = re.findall(pattern, description)[0]
    
    return rule
    

# Set your model ID
model_id = "gpt-3.5-turbo"
all_rules = []
# Process each task
for task_id in range(1, 251):
    rule = process_task_json(task_id)
    all_rules.append(rule)

with open("knowledge.json", "w") as f:
    json.dump(all_rules, f, indent=4)