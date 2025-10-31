import os
import time
import shutil
import argparse
import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from utils import (mkdir, setup_seed, convert_seconds, load_task_data, plot_loss_curve, tsne, my_chat_template, client,
                   create_task_data_lookup, get_gpu_memory_usage, load_pretrain_data_hf, post_process_for_prompting, post_process_for_y)
from tqdm import tqdm
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import openai

def train_subtask(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template):

    if args.zero_init:
        params = torch.normal(mean=0, std=1e-2, size=(1, nesy.args.latent_size), requires_grad=True, device=nesy.args.task_device, dtype=torch.float32)
    else:
        params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.float32)
    
    optimizer = torch.optim.Adam([params], lr=args.task_finetune_lr)
    keep_training = True
    test_loss_ls = []
    
    #return params, test_loss_ls
    
    training_step = 0
    while keep_training:

        for batch in subtask_train_data_loader:

            optimizer.zero_grad()
            knowledge_batch = batch["knowledge"]
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]

            if args.use_knowledge_in_task.lower() in ["hard", "soft"]:
                if args.use_chat_template:
                    x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
                else:
                    x_batch = [knowledge_batch[i] + x_batch[i] for i in range(len(x_batch))]
            else:
                if args.use_chat_template:
                    x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

            expanded_params = params.repeat_interleave(len(x_batch), dim=0)
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch) #* args.task_loss_weight + args.reg_loss_weight * params.norm(1, dim=1).mean() / args.latent_size
            task_loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % 10 == 0:
                test_loss = 0
                with torch.no_grad():
                    for batch in subtask_valid_data_loader:
                        knowledge_batch = batch["knowledge"]
                        x_batch = batch["input"]
                        x_batch = [prompt_template.format(x) for x in x_batch]
                        y_batch = batch["target"]

                        # if args.use_knowledge_in_task.lower() == "hard":
                        #     x_batch = [knowledge + x_batch[i] for i in range(len(x_batch))]

                        if args.use_knowledge_in_task.lower() in ["hard", "soft"]:
                            if args.use_chat_template:
                                x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
                            else:
                                x_batch = [knowledge + x_batch[i] for i in range(len(x_batch))]
                        else:
                            if args.use_chat_template:
                                x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
                        if args.use_knowledge_in_task.lower() == "soft":
                            knowledge_for_task_params = input_embeds.view(input_embeds.shape[0], -1)
                            params = torch.cat([params, knowledge_for_task_params], dim=1)
                            original_soft_token = nesy.args.num_soft_token
                            original_latent_size = nesy.args.latent_size
                            nesy.args.num_soft_token = original_soft_token + input_embeds.shape[1]
                            nesy.args.latent_size = params.shape[1]

                        expanded_params = params.repeat_interleave(len(x_batch), dim=0)
                        test_loss += nesy.compute_task_loss(expanded_params, x_batch, y_batch)

                        if args.use_knowledge_in_task.lower() == "soft":
                            nesy.args.num_soft_token = original_soft_token
                            nesy.args.latent_size = original_latent_size

                    test_loss /= len(subtask_valid_data_loader.dataset)
                    test_loss_ls.append(test_loss.tolist())
                    if len(test_loss_ls) > args.task_finetune_step:
                        if test_loss_ls[-1] > test_loss_ls[-2]:
                            keep_training = False
                            break
    
    return params, [0], test_loss_ls

def train_subtask_indirect(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template):

    #knowledge = "<instruction>Follow the instruction and answer the question: I do not know anything.</instruction>"
    knowledge = "<instruction>You are given an input sentence, you need to generate the output sentence based on the given input. Unfortunately, you do not know anything about what you should do about the input.</instruction>"
    knowledge_id = nesy.llm.tokenizer(knowledge, return_tensors="pt", add_special_tokens=True).input_ids.to(nesy.args.encoder_device)
    input_embeds = torch.nn.Parameter(nesy.llm.encoder_model.model.embed_tokens(knowledge_id))#.repeat(embedding.shape[0], 1, 1)

    if args.use_knowledge_in_task.lower() == "soft":
        optimizer_lr = args.lr
    else:
        optimizer_lr = args.task_finetune_lr

    optimizer = torch.optim.Adam([input_embeds], lr=optimizer_lr)
    keep_training = True
    train_loss_ls = []
    test_loss_ls = []
    
    training_step = 0
    while keep_training:

        for batch in subtask_train_data_loader:

            optimizer.zero_grad()
            knowledge_batch = batch["knowledge"]
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]

            if args.use_knowledge_in_task.lower() in ["hard", "soft"]:
                if args.use_chat_template:
                    x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
                else:
                    x_batch = [knowledge_batch[i] + x_batch[i] for i in range(len(x_batch))]
            else:
                if args.use_chat_template:
                    x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

            params, _ = nesy.encode(input_embeds)
            params = params.to(nesy.args.task_device)
            expanded_params = params.repeat_interleave(len(x_batch), dim=0)
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch) #* args.task_loss_weight + args.reg_loss_weight * params.norm(1, dim=1).mean() / args.latent_size
            task_loss.backward()
            optimizer.step()
            training_step += 1
            train_loss_ls.append(task_loss.tolist())

            if training_step % 10 == 0:
                test_loss = 0
                with torch.no_grad():
                    for batch in subtask_valid_data_loader:
                        knowledge_batch = batch["knowledge"]
                        x_batch = batch["input"]
                        x_batch = [prompt_template.format(x) for x in x_batch]
                        y_batch = batch["target"]

                        # if args.use_knowledge_in_task.lower() == "hard":
                        #     x_batch = [knowledge + x_batch[i] for i in range(len(x_batch))]

                        if args.use_knowledge_in_task.lower() in ["hard", "soft"]:
                            if args.use_chat_template:
                                x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
                            else:
                                x_batch = [knowledge + x_batch[i] for i in range(len(x_batch))]
                        else:
                            if args.use_chat_template:
                                x_batch = [nesy.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

                        params, _ = nesy.encode(input_embeds)

                        if args.use_knowledge_in_task.lower() == "soft":
                            knowledge_for_task_params = input_embeds.view(input_embeds.shape[0], -1)
                            params = torch.cat([params, knowledge_for_task_params], dim=1)
                            original_soft_token = nesy.args.num_soft_token
                            original_latent_size = nesy.args.latent_size
                            nesy.args.num_soft_token = original_soft_token + input_embeds.shape[1]
                            nesy.args.latent_size = params.shape[1]

                        params = params.to(nesy.args.task_device)
                        expanded_params = params.repeat_interleave(len(x_batch), dim=0)
                        test_loss += nesy.compute_task_loss(expanded_params, x_batch, y_batch)

                        if args.use_knowledge_in_task.lower() == "soft":
                            nesy.args.num_soft_token = original_soft_token
                            nesy.args.latent_size = original_latent_size

                    test_loss /= len(subtask_valid_data_loader.dataset)
                    test_loss_ls.append(test_loss.tolist())
                    if len(test_loss_ls) > args.task_finetune_step:
                        if test_loss_ls[-1] > test_loss_ls[-2]:
                            keep_training = False
                            break

    params, _ = nesy.encode(input_embeds)
    params = params.to(nesy.args.task_device)
    return params, train_loss_ls, test_loss_ls

def test_symbolic2neural(args, epoch, data_loader, nesy, prompt_template, evaluater, log, name):
    
    log.writelines(f"epoch {epoch} \n")

    start_time = time.time()
    num_correct = 0
    num_test = 0

    for batch in data_loader:
        
        with torch.no_grad():
            knowledge_batch = batch["knowledge"]
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]
            
            results = nesy.eval_task(knowledge_batch, x_batch, y_batch, evaluater)
            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct += result["score"]
                num_test += 1
                log.flush()
                

    accuracy = num_correct / num_test
    log.writelines(f"symbolic2neural accuracy on {name} samples: {accuracy} \n")
    end_time = time.time()
    cost_time = convert_seconds(end_time-start_time)
    log.writelines(f"symbolic2neural validation on {name} finished, time cost {cost_time} \n")
    log.flush()

def test_neural2symbolic(args, epoch, test_data, nesy, prompt_template, evaluater, log, name):

    log.writelines(f"epoch {epoch} \n")

    start_time = time.time()
    all_tasks_ids = list(set([sample["sub_task_id"] for sample in test_data]))

    #all_tasks_ids = random.sample(all_tasks_ids, 10)

    num_correct_symbolic = 0
    num_test_symbolic = 0

    for task_id in all_tasks_ids:

        # subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        # subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

        subtask_data = [data for data in test_data if data["sub_task_id"] == task_id]
        subtask_train_data = subtask_data#[:-1]
        subtask_valid_data = subtask_data[-1:]

        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_valid_data_loader = DataLoader(subtask_valid_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_valid_data[0]["knowledge"]

        knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt", add_special_tokens=True).input_ids.to(nesy.args.encoder_device)
        #encoded_latent = [nesy.reparameterize(*nesy.encode(knowledge_ids)) for i in range(num_samples)]
        #randomn_latent = [torch.randn([1, nesy.args.latent_size]) for i in range(num_samples)]

        if args.indirect_finetune:
            trained_params, train_loss_ls, test_loss_ls = train_subtask_indirect(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template)
        else:
            trained_params, train_loss_ls, test_loss_ls = train_subtask(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template)

        with torch.no_grad():

            if args.method == "vaeflow":
                trained_latent = trained_params.to(nesy.args.flow_device)
                trained_latent = nesy.flow_backward(trained_params).to(nesy.args.decoder_device)
            else:
                if args.nf:
                    trained_latent = nesy.flow_backward(trained_params.to(nesy.args.flow_device)).to(nesy.args.decoder_device)
                else:
                    trained_latent = trained_params.to(nesy.args.decoder_device)

            if nesy.args.use_instance_in_decoder:
                batch = random.choice(subtask_train_data_loader.dataset)
                x = batch["input"]
                y = batch["target"]
                instance_text = f"input: {x}, target: {y}. This task is to:"
                print(instance_text)
                instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
            else:
                instance_ids = None
                # instance_text = f"This task is to:"
                # print(instance_text)
                # instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
            
            predicted_knowledge = nesy.predict_knowledge(trained_latent, sample_from_guassian=False, instance=instance_ids)
            #encoded_params = encoded_latent[i].to(nesy.args.decoder_device)
            #encode_decode_knowledge = nesy.sample(encoded_params, sample_from_guassian=False)

            reconstructed_latent = nesy.encode(knowledge_ids)[0].to(args.decoder_device)
            reconstructed_knowledge = nesy.predict_knowledge(reconstructed_latent, sample_from_guassian=False, instance=instance_ids)

        log.writelines(f"prediction on {name} subtask {task_id}: \n")
        log.writelines(f"subtask train loss: {str(train_loss_ls)} \n")
        log.writelines(f"subtask test loss: {str(test_loss_ls)} \n")
        result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)
        log.writelines(f"{json.dumps(result, indent=4)}\n")
        log.writelines(f"reconstructed_knowledge: {reconstructed_knowledge} \n")
        num_correct_symbolic += result["score"]
        # result = nesy.eval_knowledge(knowledge, encode_decode_knowledge, evaluater)
        # log.writelines(f"{json.dumps(result, indent=4)}\n")
        num_test_symbolic += 1
        log.flush()
        

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"neural2symbolic accuracy on {name} samples: {accuracy} \n")
    end_time = time.time()
    cost_time = convert_seconds(end_time-start_time)
    log.writelines(f"neural2symbolic validation on {name} finished, time cost {cost_time} \n")
    log.flush()

def test_neural_task(args, seen_task_train_data_loader, seen_task_test_data_loader, unseen_task_test_data_loader, nesy, prompt_template, evaluater, log, method):

    log.writelines(f"neural task testing for method: {method} \n")
    log.flush()

    num_correct_neural = 0
    num_test_neural = 0

    if method == "finetuning":

        params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.float32)
        if args.use_trainable_task_model:
            optimizer = torch.optim.Adam(
                [{'params': params, 'lr': args.task_finetune_lr}, 
                 {'params': nesy.llm.task_model.parameters(), 'lr': args.lr}], 
                lr=args.task_finetune_lr
            )
        else:
            optimizer = torch.optim.Adam([params], lr=args.task_finetune_lr)
        keep_training = True
        test_loss_ls = []
        
        while keep_training:

            for i, batch in tqdm(enumerate(seen_task_train_data_loader)):

                if i % 100 == 0:
                    test_loss = 0
                    with torch.no_grad():
                        for batch in seen_task_test_data_loader:
                            knowledge_batch = batch["knowledge"]
                            batch_size = len(knowledge_batch)
                            x_batch = batch["input"]
                            x_batch = [prompt_template.format(x) for x in x_batch]
                            y_batch = batch["target"]
                            input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
                            input_batch = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
                            expanded_params = params.repeat_interleave(len(input_batch), dim=0)
                            test_loss += nesy.compute_task_loss(expanded_params, input_batch, y_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step*3:
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                knowledge_batch = batch["knowledge"]
                batch_size = len(knowledge_batch)
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]
                input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
                input_batch = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
                expanded_params = params.repeat_interleave(len(input_batch), dim=0)
                task_loss = nesy.compute_task_loss(expanded_params, input_batch, y_batch)
                task_loss.backward()
                optimizer.step()

    # start testing neural task
    with torch.no_grad():

        for batch in seen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]

            if method == "icl":
                input_message = [[
                                  {"role": "system", "content": "<instruction>Translate the input text into English.</instruction>"},
                                  {"role": "user", "content": "<input>你好，世界。<\input>"},
                                  {"role": "assistant", "content": "<output>Hello, world.</output>"},
                                  {"role": "system", "content": knowledge_batch[i]}, 
                                  {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            else:
                input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            input_text = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)

            # input_batch = [knowledge_prompt.format(knowledge_batch[i], x_batch[i]) for i in range(batch_size)]
            # input_ids = nesy.llm.tokenizer(input_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)
            
            if method in ["prompting", "icl"]:
                y_pred = nesy.llm.predict_task(input_ids)
                y_pred = [post_process_for_prompting(y) for y in y_pred]
            elif method == "finetuning":
                if args.fuse_method in ["delta", "lora"]:
                    new_task_parameters = nesy.llm.allocate(params[0])
                    y_pred = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(len(input_text), dim=0)
                    y_pred = nesy.llm.predict_task(input_ids, expanded_params)

            y_pred = [y.split("\n")[0] for y in y_pred]

            results = [{
                "knowledge": knowledge_batch[i],
                "x": x_batch[i],
                "y_true": y_batch[i],
                "y_pred": y_pred[i],
                #"score": evaluater(y_pred[i], y_batch[i])
                "score": evaluater(y_pred[i], y_batch[i], x_batch[i], knowledge_batch[i])
                } for i in range(len(x_batch))]

            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_neural += result["score"]
                num_test_neural += 1
                log.flush()

    accuracy = num_correct_neural / num_test_neural
    log.writelines(f"neural seen task accuracy of method {method}: {accuracy} \n")
    log.flush()

    with torch.no_grad():

        for batch in unseen_task_test_data_loader:
            knowledge_batch = batch["knowledge"]
            batch_size = len(knowledge_batch)
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]

            if method == "icl":
                input_message = [[
                                  {"role": "system", "content": "<instruction>Translate the input text into English.</instruction>"},
                                  {"role": "user", "content": "<input>你好，世界。<\input>"},
                                  {"role": "assistant", "content": "<output>Hello, world.</output>"},
                                  {"role": "system", "content": knowledge_batch[i]}, 
                                  {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            else:
                input_message = [[{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}] for i in range(len(x_batch))]
            input_text = [nesy.llm.tokenizer.apply_chat_template(input_message[i], tokenize=False) for i in range(len(input_message))]
            input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)
            
            if method in ["prompting", "icl"]:
                y_pred = nesy.llm.predict_task(input_ids)
                y_pred = [post_process_for_prompting(y) for y in y_pred]
            elif method == "finetuning":
                if args.fuse_method in ["delta", "lora"]:
                    new_task_parameters = nesy.llm.allocate(params[0])
                    y_pred = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(len(input_text), dim=0)
                    y_pred = nesy.llm.predict_task(input_ids, expanded_params)

            y_pred = [y.split("\n")[0] for y in y_pred]

            results = [{
                "knowledge": knowledge_batch[i],
                "x": x_batch[i],
                "y_true": y_batch[i],
                "y_pred": y_pred[i],
                #"score": evaluater(y_pred[i], y_batch[i])
                "score": evaluater(y_pred[i], y_batch[i], x_batch[i], knowledge_batch[i])
                } for i in range(len(x_batch))]

            for result in results:
                log.writelines(f"{json.dumps(result, indent=4)}\n")
                num_correct_neural += result["score"]
                num_test_neural += 1
                log.flush()

    accuracy = num_correct_neural / num_test_neural
    log.writelines(f"neural unseen task accuracy of method {method}: {accuracy} \n")
    log.flush()

def test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, prompt_template, evaluater, log, method):

    log.writelines(f"symbolic task testing for method: {method} \n")
    log.flush()

    #sys_prompt = "Given the following input and output pairs, please infer their shared instruction."

    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "\nThe instruction was: "
    
    if method in ["finetuning"]:

        seen_train_data = seen_train_data_loader.dataset
        seen_test_data = seen_test_data_loader.dataset
        seen_tasks_ids = list(set([sample["sub_task_id"] for sample in seen_train_data]))
        seen_train_data_induction = []
        seen_test_data_induction = []

        for task_id in seen_tasks_ids:

            seen_subtask_train_data = [data for data in seen_train_data if data["sub_task_id"] == task_id]
            seen_subtask_test_data = [data for data in seen_test_data if data["sub_task_id"] == task_id]
            knowledge = seen_subtask_train_data[0]["knowledge"]

            if len(seen_subtask_train_data) >= args.test_sample_num:
                for _ in range(10):
                    io_sample_train = random.sample(seen_subtask_train_data, args.test_sample_num)
                    io_text_train = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in io_sample_train])
                    seen_train_data_induction.append({
                        "knowledge": knowledge,
                        "io_text": io_text_train
                        })
            if len(seen_subtask_test_data) > 0:
                io_sample_test = random.sample(seen_subtask_test_data, args.test_sample_num)
                io_text_test = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in io_sample_test])
                seen_test_data_induction.append({
                    "knowledge": knowledge,
                    "io_text": io_text_test
                    })
        
        seen_task_train_data_loader = DataLoader(seen_train_data_induction, batch_size=args.batch_size//4, shuffle=True)
        seen_task_test_data_loader = DataLoader(seen_test_data_induction, batch_size=args.batch_size//4, shuffle=True)

        params = torch.randn(size=[1, nesy.args.latent_size], requires_grad=True, device=nesy.args.task_device, dtype=torch.float32)
        if args.use_trainable_task_model:
            optimizer = torch.optim.Adam(
                [{'params': params, 'lr': args.task_finetune_lr}, 
                 {'params': nesy.llm.task_model.parameters(), 'lr': args.lr}], 
                lr=args.task_finetune_lr
            )
        else:
            optimizer = torch.optim.Adam([params], lr=args.task_finetune_lr)        
        keep_training = True
        test_loss_ls = []
        
        while keep_training:

            for i, batch in tqdm(enumerate(seen_task_train_data_loader)):

                if i % 100 == 0:
                    test_loss = 0
                    with torch.no_grad():
                        for batch in seen_task_test_data_loader:
                            knowledge_batch = batch["knowledge"]
                            batch_size = len(knowledge_batch)
                            #io_batch = [prompt.format(batch["io_text"][i]) for i in range(batch_size)]
                            #io_message = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": batch["io_text"][i]}] for i in range(batch_size)]
                            #io_batch = [nesy.llm.tokenizer.apply_chat_template(io_message[i], tokenize=False) for i in range(batch_size)]
                            io_batch = [fore_prompt + batch["io_text"][i] + post_prompt for i in range(batch_size)]
                            expanded_params = params.repeat_interleave(len(io_batch), dim=0)
                            test_loss += nesy.compute_task_loss(expanded_params, io_batch, knowledge_batch)
                        test_loss /= len(seen_task_test_data_loader)
                        test_loss_ls.append(test_loss.tolist())
                        log.writelines(f"{test_loss.tolist()}\n")
                        log.flush()
                        if len(test_loss_ls) > args.task_finetune_step:
                            if test_loss_ls[-1] > test_loss_ls[-2]:
                                keep_training = False
                                break

                optimizer.zero_grad()
                knowledge_batch = batch["knowledge"]
                batch_size = len(knowledge_batch)
                #io_message = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": batch["io_text"][i]}] for i in range(batch_size)]
                #io_batch = [nesy.llm.tokenizer.apply_chat_template(io_message[i], tokenize=False) for i in range(batch_size)]
                io_batch = [fore_prompt + batch["io_text"][i] + post_prompt for i in range(batch_size)]
                expanded_params = params.repeat_interleave(len(io_batch), dim=0)
                task_loss = nesy.compute_task_loss(expanded_params, io_batch, knowledge_batch)
                task_loss.backward()
                optimizer.step()

    seen_test_data = seen_test_data_loader.dataset
    seen_tasks_ids = list(set([sample["sub_task_id"] for sample in seen_test_data]))

    num_correct_symbolic = 0
    num_test_symbolic = 0

    for task_id in seen_tasks_ids:

        seen_subtask_data = [data for data in seen_test_data if data["sub_task_id"] == task_id]
        knowledge = seen_subtask_data[0]["knowledge"]

        # start testing symbolic task
        with torch.no_grad():
            
            obeserved_samples = random.sample(seen_subtask_data, args.test_sample_num)

            if method == "icl":
                obeserved_text = "\n".join([f"{data['input']}{data['target']}" for data in obeserved_samples])
                input_message = [{"role": "system", "content": sys_prompt}, 
                                 {"role": "user", \
                                  "content": "<input>你好，世界。</input><output>Hello, world.</output>\n \
                                              <input>可以介绍一下什么是机器学习吗。</input><output>Can you explain what machine learning is?</output>\n \
                                              <input>我还不是很明白。</input><output>I'm still not very clear.</output>\n \
                                              <input>我需要一个翻译工具。</input><output>I need a translation tool.</output>\n \
                                              <input>你只需要一个大语言模型。</input><output>A large language model is all you need.</output>\n"},
                                 {"role": "assistant", "content": "<instruction>Translate the input text into English.</instruction>"},
                                 {"role": "user", "content": obeserved_text}]
            else:
                obeserved_text = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in obeserved_samples])
                input_message = fore_prompt + obeserved_text + post_prompt
                #input_message = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obeserved_text}]
            #input_text = nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False)
            input_ids = nesy.llm.tokenizer(input_message, return_tensors="pt").input_ids.to(nesy.args.task_device)

            if method in ["prompting", "icl"]:
                predicted_knowledge = nesy.llm.predict_task(input_ids)#[0]
                #predicted_ids = nesy.llm.task_model.generate(input_ids, max_new_tokens=nesy.args.max_token)
                #predicted_knowledge = nesy.llm.tokenizer.decode(predicted_ids[0][len(input_ids[0]):], skip_special_tokens=True)
                predicted_knowledge = post_process_for_prompting(predicted_knowledge)
            elif method in ["finetuning"]:
                if args.fuse_method in ["delta", "lora"]:
                    new_task_parameters = nesy.llm.allocate(params[0])
                    predicted_knowledge = nesy.llm.predict_task(input_ids, new_task_parameters)
                    # predicted_knowledge = nesy.predict_knowledge(params.to(nesy.args.decoder_device), sample_from_guassian=False, instance=None)

                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(input_ids.shape[0], dim=0)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, expanded_params)

            if type(predicted_knowledge) == list:
                predicted_knowledge = predicted_knowledge[0]
            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)

            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            num_test_symbolic += 1
            log.flush()

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"symbolic seen task accuracy of method {method}: {accuracy} \n")
    log.flush()

    unseen_test_data = unseen_test_data_loader.dataset
    unseen_tasks_ids = list(set([sample["sub_task_id"] for sample in unseen_test_data]))

    num_correct_symbolic = 0
    num_test_symbolic = 0

    for task_id in unseen_tasks_ids:

        unseen_subtask_data = [data for data in unseen_test_data if data["sub_task_id"] == task_id]
        knowledge = unseen_subtask_data[0]["knowledge"]

        # start testing symbolic task
        with torch.no_grad():
            
            obeserved_samples = random.sample(unseen_subtask_data, args.test_sample_num)

            if method == "icl":
                obeserved_text = "\n".join([f"{data['input']}{data['target']}" for data in obeserved_samples])
                input_message = [{"role": "system", "content": sys_prompt}, 
                                 {"role": "user", \
                                  "content": "<input>你好，世界。</input><output>Hello, world.</output>\n \
                                              <input>可以介绍一下什么是机器学习吗。</input><output>Can you explain what machine learning is?</output>\n \
                                              <input>我还不是很明白。</input><output>I'm still not very clear.</output>\n \
                                              <input>我需要一个翻译工具。</input><output>I need a translation tool.</output>\n \
                                              <input>你只需要一个大语言模型。</input><output>A large language model is all you need.</output>\n"},
                                 {"role": "assistant", "content": "<instruction>Translate the input text into English.</instruction>"},
                                 {"role": "user", "content": obeserved_text}]
            else:
                obeserved_text = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in obeserved_samples])
                #input_message = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": obeserved_text}]
                input_message = fore_prompt + obeserved_text + post_prompt
            #input_text = nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False)
            input_ids = nesy.llm.tokenizer(input_message, return_tensors="pt").input_ids.to(nesy.args.task_device)

            if method in ["prompting", "icl"]:
                predicted_knowledge = nesy.llm.predict_task(input_ids)#[0]
                #predicted_ids = nesy.llm.task_model.generate(input_ids, max_new_tokens=nesy.args.max_token)
                #predicted_knowledge = nesy.llm.tokenizer.decode(predicted_ids[0][len(input_ids[0]):], skip_special_tokens=True)
                predicted_knowledge = post_process_for_prompting(predicted_knowledge)
            elif method in ["finetuning"]:
                if args.fuse_method in ["delta", "lora"]:
                    new_task_parameters = nesy.llm.allocate(params[0])
                    predicted_knowledge = nesy.llm.predict_task(input_ids, new_task_parameters)
                elif args.fuse_method == "p-tuning":
                    expanded_params = params.repeat_interleave(input_ids.shape[0], dim=0)
                    predicted_knowledge = nesy.llm.predict_task(input_ids, expanded_params)

            if type(predicted_knowledge) == list:
                predicted_knowledge = predicted_knowledge[0]
            result = nesy.eval_knowledge(knowledge, predicted_knowledge, evaluater)

            log.writelines(f"{json.dumps(result, indent=4)}\n")
            num_correct_symbolic += result["score"]
            num_test_symbolic += 1
            log.flush()

    accuracy = num_correct_symbolic / num_test_symbolic
    log.writelines(f"symbolic unseen task accuracy of method {method}: {accuracy} \n")
    log.flush()

def iterative_inference(args, unseen_train_data_loader, unseen_test_data_loader, nesy, prompt_template, evaluater, log):

    unseen_train_data = unseen_train_data_loader.dataset
    unseen_test_data = unseen_test_data_loader.dataset
    unseen_tasks_ids = list(set([sample["sub_task_id"] for sample in unseen_train_data]))
    test_task_id = random.choice(unseen_tasks_ids)

    test_subtask_train_data = [data for data in unseen_train_data if data["sub_task_id"] == test_task_id]
    test_subtask_test_data = [data for data in unseen_test_data if data["sub_task_id"] == test_task_id]
    test_subtask_train_dataloader = DataLoader(test_subtask_train_data, batch_size=args.batch_size, shuffle=True)
    test_subtask_test_dataloader = DataLoader(test_subtask_test_data, batch_size=args.batch_size, shuffle=True)
    knowledge_groundtruth = test_subtask_train_data[0]["knowledge"]

    log.writelines(f"selected task: {test_task_id} \n")
    log.writelines(f"x sample: {test_subtask_train_data[0]['input']} \n")
    log.writelines(f"y sample: {test_subtask_train_data[0]['target']} \n")
    log.writelines(f"knowledge_groundtruth: {knowledge_groundtruth} \n")
    log.flush()

    def test():

        num_correct = 0
        num_test = 0
        for batch in test_subtask_test_dataloader:
            
            with torch.no_grad():
                knowledge_batch = [knowledge] * len(batch["input"])
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                results = nesy.eval_task(knowledge_batch, x_batch, y_batch, evaluater, knowledge_groundtruth=[knowledge_groundtruth] * len(x_batch))
                for result in results:
                    log.writelines(f"{json.dumps(result, indent=4)}\n")
                    num_correct += result["score"]
                    num_test += 1
                    log.flush()

        accuracy = num_correct / num_test

        return accuracy

    #迭代次数
    knowledge = "<instruction>I don't know anything.</instruction>"

    for iter_num in range(10):

        log.writelines(f"neural inference iter {iter_num} with knowledge {knowledge} \n")
        accuracy = test()
        log.writelines(f"neural accuracy: {accuracy} \n")
        log.flush()

        log.writelines(f"neural inference iter {iter_num} with finetuning on task data \n")
        if args.use_knowledge_in_task.lower() == "hard":
            knowledge_id = nesy.llm.tokenizer(knowledge, return_tensors="pt", add_special_tokens=True).input_ids.to(nesy.args.encoder_device)
        else:
            knowledge_id = nesy.llm.tokenizer(knowledge, return_tensors="pt", add_special_tokens=False).input_ids.to(nesy.args.encoder_device)
        input_embeds = torch.nn.Parameter(nesy.llm.encoder_model.model.embed_tokens(knowledge_id))#.repeat(embedding.shape[0], 1, 1)

        if args.use_knowledge_in_task.lower() == "soft":
            optimizer_lr = args.lr
        else:
            optimizer_lr = args.task_finetune_lr
        optimizer = torch.optim.Adam([input_embeds], lr=optimizer_lr)
        for batch in test_subtask_train_dataloader:
            knowledge_batch = [knowledge] * len(batch["input"])
            x_batch = batch["input"]
            y_batch = batch["target"]
            optimizer.zero_grad()
            x_batch = batch["input"]
            x_batch = [prompt_template.format(x) for x in x_batch]
            y_batch = batch["target"]
            params, _ = nesy.encode(input_embeds)
            params = params.to(nesy.args.task_device)
            expanded_params = params.repeat_interleave(len(x_batch), dim=0)
            task_loss = nesy.compute_task_loss(expanded_params, x_batch, y_batch) #* args.task_loss_weight + args.reg_loss_weight * params.norm(1, dim=1).mean() / args.latent_size
            task_loss.backward()
            optimizer.step()

        accuracy = test()
        log.writelines(f"neural accuracy: {accuracy} \n")
        log.flush()

        log.writelines(f"induction iter {iter_num} \n")
        params, _ = nesy.encode(input_embeds)
        if nesy.args.use_instance_in_decoder:
            batch = random.choice(test_subtask_train_dataloader.dataset)
            x = batch["input"]
            y = batch["target"]
            instance_text = f"input: {x}, target: {y}. This task is to:"
            print(instance_text)
            instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
        else:
            instance_ids = None
            # instance_text = f"This task is to:"
            # print(instance_text)
            # instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
        knowledge = nesy.predict_knowledge(params.to(args.decoder_device), sample_from_guassian=False, instance=instance_ids)
        log.writelines(f"induced knowledge: {knowledge} \n")
        log.flush()

def refinement_inference(args, train_data_loader, test_data_loader, nesy, prompt_template, evaluater, log, name, load=False):

    if not load:

        train_data = train_data_loader.dataset
        test_data = test_data_loader.dataset

        # num_correct = 0
        # num_test = 0

        # for batch in test_data_loader:
            
        #     with torch.no_grad():
        #         knowledge_batch = batch["knowledge"]
        #         x_batch = batch["input"]
        #         x_batch = [prompt_template.format(x) for x in x_batch]
        #         y_batch = batch["target"]
                
        #         results = nesy.eval_task(knowledge_batch, x_batch, y_batch, evaluater)
        #         for result in results:
        #             log.writelines(f"{json.dumps(result, indent=4)}\n")
        #             num_correct += result["score"]
        #             num_test += 1
        #             log.flush()

        # accuracy = num_correct / num_test
        # log.writelines(f"groundtruth knowledge injection accuracy on {name} samples: {accuracy} \n")
        # log.flush()

        all_groundtruth_knowledge = []
        all_encoded_groundtruth_knowledge = []
        all_trained_params = []
        all_encoded_induced_knowledge = []

        all_tasks_ids = list(set([sample["sub_task_id"] for sample in train_data]))

        num_correct_finetune = 0
        num_test_finetune = 0
        num_correct_refine = 0
        num_test_refine = 0

        for task_id in all_tasks_ids:

            # subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
            # subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

            subtask_data = [data for data in train_data if data["sub_task_id"] == task_id]
            subtask_train_data = subtask_data#[:-1]
            subtask_valid_data = subtask_data[-1:]
            subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

            subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
            subtask_valid_data_loader = DataLoader(subtask_valid_data, batch_size=args.batch_size, shuffle=True)
            subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
            knowledge = subtask_valid_data[0]["knowledge"]

            all_groundtruth_knowledge.append(knowledge)

            with torch.no_grad():

                knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
                encoded_mean, encoded_logvar = nesy.encode(knowledge_ids)
                all_encoded_groundtruth_knowledge.append(encoded_mean.to(torch.float16).cpu().numpy())

            if args.indirect_finetune:
                trained_params, train_loss_ls, test_loss_ls = train_subtask_indirect(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template)
            else:
                trained_params, train_loss_ls, test_loss_ls = train_subtask(args, nesy, subtask_train_data_loader, subtask_valid_data_loader, prompt_template)

            all_trained_params.append(trained_params.to(torch.float16).detach().cpu().numpy())

            with torch.no_grad():

                if args.method == "vaeflow":
                    trained_latent = trained_params.to(nesy.args.flow_device)
                    trained_latent = nesy.flow_backward(trained_params).to(nesy.args.decoder_device)
                else:
                    if args.nf:
                        trained_latent = nesy.flow_backward(trained_params.to(nesy.args.flow_device)).to(nesy.args.decoder_device)
                    else:
                        trained_latent = trained_params.to(nesy.args.decoder_device)

                if nesy.args.use_instance_in_decoder:
                    batch = random.choice(subtask_train_data_loader.dataset)
                    x = batch["input"]
                    y = batch["target"]
                    instance_text = f"input: {x}, target: {y}. This task is to:"
                    instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
                else:
                    instance_ids = None
                    # instance_text = f"This task is to:"
                    # print(instance_text)
                    # instance_ids = nesy.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.decoder_device)
                    
                predicted_knowledge = nesy.predict_knowledge(trained_latent, sample_from_guassian=False, instance=instance_ids)

                predicted_knowledge_ids = nesy.llm.tokenizer(predicted_knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
                encoded_predicted_mean, encoded_predicted_logvar = nesy.encode(predicted_knowledge_ids)
                all_encoded_induced_knowledge.append(encoded_predicted_mean.to(torch.float16).cpu().numpy())

                for batch in subtask_test_data_loader:
                    
                    with torch.no_grad():
                        knowledge_batch = [predicted_knowledge] * len(batch["input"])
                        x_batch = batch["input"]
                        x_batch = [prompt_template.format(x) for x in x_batch]
                        y_batch = batch["target"]

                        results = nesy.eval_task(knowledge_batch, x_batch, y_batch, evaluater)
                        for result in results:
                            log.writelines(f"{json.dumps(result, indent=4)}\n")
                            num_correct_refine += result["score"]
                            num_test_refine += 1
                            log.flush()

                        x_id = nesy.llm.tokenizer(x_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(nesy.args.task_device)
                        y_pred = nesy.llm.predict_task(x_id, trained_params.repeat_interleave(len(x_batch), dim=0))
                        results = [
                            {
                                "knowledge": knowledge,
                                "type": "trained",
                                "x": x_batch[i],
                                "y_true": y_batch[i],
                                "y_pred": y_pred[i],
                                #"score": evaluater(y_pred[i], y_batch[i])
                                "score": evaluater(y_pred[i], y_batch[i], x_batch[i], knowledge)
                            }
                            for i in range(len(x_batch))
                        ]

                        for result in results:
                            log.writelines(f"{json.dumps(result, indent=4)}\n")
                            num_correct_finetune += result["score"]
                            num_test_finetune += 1
                            log.flush()

        accuracy_finetune = num_correct_finetune / num_test_finetune
        accuracy_refine = num_correct_refine / num_test_refine
        log.writelines(f"finetune knowledge injection accuracy on {name} samples: {accuracy_finetune} \n")
        log.writelines(f"refined knowledge injection accuracy on {name} samples: {accuracy_refine} \n")
        log.flush()

        groundtruth_latent = np.concatenate(all_encoded_groundtruth_knowledge, axis=0)
        trained_latent = np.concatenate(all_trained_params, axis=0)
        induced_latent = np.concatenate(all_encoded_induced_knowledge, axis=0)
        
        # 使用 TSNE 降维
        n_knowledge = len(all_encoded_groundtruth_knowledge)
        all_latent = np.concatenate([groundtruth_latent, trained_latent, induced_latent], axis=0)
        tsne = TSNE(n_components=2, perplexity=5)
        tsne_result = tsne.fit_transform(all_latent)

        # 创建 DataFrame 用于绘图
        df_groundtruth = pd.DataFrame(tsne_result[:n_knowledge], columns=['x', 'y'])
        df_groundtruth['type'] = 'Groundtruth'
        df_groundtruth['knowledge'] = all_groundtruth_knowledge
        df_trained = pd.DataFrame(tsne_result[n_knowledge:n_knowledge*2], columns=['x', 'y'])
        df_trained['type'] = 'Trained'
        df_trained['knowledge'] = all_groundtruth_knowledge
        df_induced = pd.DataFrame(tsne_result[n_knowledge*2:], columns=['x', 'y'])
        df_induced['type'] = 'Induced'
        df_induced['knowledge'] = all_groundtruth_knowledge

        # 合并数据
        df_combined = pd.concat([df_groundtruth, df_trained, df_induced], ignore_index=True)

        df_combined.to_csv(f"{args.exp_dir}/{name}_latent.csv")

    else:

        df_combined = pd.read_csv(f"{args.exp_dir}/{name}_latent.csv")

    x_range = df_combined['x'].min()-20, df_combined['x'].max()+20
    y_range = df_combined['y'].min()-20, df_combined['y'].max()+20

    # 绘制第一张图：只显示 all_encoded_groundtruth_knowledge
    fig2 = px.scatter(
        df_combined,
        x='x',
        y='y',
        hover_data={'knowledge': True, 'x': False, 'y': False},
        color='type',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],  # 不同颜色区分
        template='plotly_white'
    )

    # 更新图形布局
    fig2.update_layout(
        xaxis=dict(title=None, showticklabels=False, showgrid=True, gridcolor='white', 
                   zeroline=False, range=x_range, autorange=False),
        yaxis=dict(title=None, showticklabels=False, showgrid=True, gridcolor='white', 
                   zeroline=False, range=y_range, autorange=False),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)'  # 半透明白色背景
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',  # 改为与plot_bgcolor相同的颜色
        margin=dict(t=0, r=0, b=0, l=0)  # 去掉外侧白框
    )

    # 保存第二张图
    fig2.write_html(f"{args.exp_dir}/{name}_combined_latent.html")
    fig2.write_image(f"{args.exp_dir}/{name}_combined_latent.pdf")

    # 绘制第一张图：只显示 all_encoded_groundtruth_knowledge 
    fig1 = px.scatter(
        df_groundtruth,
        x='x',
        y='y',
        hover_data={'knowledge': True, 'x': False, 'y': False},
        color='type',
        color_discrete_sequence=['#0052CC'],
        template='plotly_white'
    )

    # 更新图形布局，使用第二张图的坐标范围
    fig1.update_layout(
        xaxis=dict(title=None, showticklabels=False, showgrid=True, gridcolor='white', 
                   zeroline=False, range=x_range, autorange=False),
        yaxis=dict(title=None, showticklabels=False, showgrid=True, gridcolor='white', 
                   zeroline=False, range=y_range, autorange=False),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)'  # 半透明白色背景
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',  # 改为与plot_bgcolor相同的颜色
        margin=dict(t=0, r=0, b=0, l=0)  # 去掉外侧白框
    )

    # 保存第一张图
    fig1.write_html(f"{args.exp_dir}/{name}_groundtruth_latent.html")
    fig1.write_image(f"{args.exp_dir}/{name}_groundtruth_latent.pdf")

def icl_inference(args, train_data_loader, test_data_loader, nesy, prompt_template, evaluater, log, name):

    train_data = train_data_loader.dataset
    test_data = test_data_loader.dataset

    all_tasks_ids = list(set([sample["sub_task_id"] for sample in train_data]))

    num_correct = 0
    num_test = 0

    for task_id in all_tasks_ids:

        # subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
        # subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

        subtask_data = [data for data in train_data if data["sub_task_id"] == task_id]
        subtask_train_data = subtask_data#[:-1]
        subtask_valid_data = subtask_data[-1:]
        subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

        subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
        subtask_valid_data_loader = DataLoader(subtask_valid_data, batch_size=args.batch_size, shuffle=True)
        subtask_test_data_loader = DataLoader(subtask_test_data, batch_size=args.batch_size, shuffle=True)
        knowledge = subtask_valid_data[0]["knowledge"]

        if args.method == "instruction_induction":
            fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
            post_prompt = "\nThe instruction was: "

            with torch.no_grad():
                
                obeserved_samples = random.sample(subtask_train_data, 5)
                obeserved_text = "\n".join([f"Input: {data['input']}. Output: {data['target']}." for data in obeserved_samples])

                input_message = [{"role": "system", "content": "Given the following input and output pairs, please directly output their shared instruction."}, 
                                 {"role": "user", \
                                  "content": "Input: 你好，世界。Output: Hello, world.\n \
                                              Input: 可以介绍一下什么是机器学习吗。Output: Can you explain what machine learning is?\n \
                                              Input: 我还不是很明白。Output: I'm still not very clear.\n \
                                              Input: 我需要一个翻译工具。Output: I need a translation tool.\n \
                                              Input: 你只需要一个大语言模型。Output: A large language model is all you need.\n"},
                                 {"role": "assistant", "content": "Translate the input text into English."},
                                 {"role": "user", "content": obeserved_text}]
                # input_text = nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False)
                # input_text = fore_prompt + obeserved_text + post_prompt
                # input_ids = nesy.llm.tokenizer(input_text, return_tensors="pt").input_ids.to(nesy.args.task_device)

                #for _ in range(5):
                #predicted_knowledge = nesy.llm.predict_task(input_ids, sample=True)[0]
                #predicted_knowledge = post_process_for_prompting(predicted_knowledge)
                predicted_knowledge = client.chat.completions.create(model="gpt-4o-mini", messages=input_message, temperature=0.5).choices[0].message.content

                print(predicted_knowledge)

        for batch in subtask_test_data_loader:
            
            with torch.no_grad():

                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                obeserved_samples = random.sample(subtask_train_data, args.test_sample_num)

                if args.method == "instruction_induction":
                    input_message = [{"role": "system", "content": predicted_knowledge}]
                else:
                    input_message = []
                for obeserved_sample in obeserved_samples:
                    input_message.append({"role": "user", "content": obeserved_sample["input"]})
                    input_message.append({"role": "assistant", "content": obeserved_sample["target"]})
                input_batch = [input_message + [{"role": "user", "content": x}] for x in x_batch]
                input_text = [nesy.llm.tokenizer.apply_chat_template(input_message, tokenize=False) for input_message in input_batch]

                input_text = [text.replace("<input>", "").replace("</input>", "").replace("<output>", "").replace("</output>", "") for text in input_text]
                #input_message = "\n".join([f"{data['input']} {data['target']}" for data in obeserved_samples])
                #input_text = [input_message + "\n" + x for x in x_batch]
                input_ids, attention_mask = nesy.llm.tokenizer(input_text, return_tensors="pt", padding="longest").values()
                input_ids = input_ids.to(nesy.args.task_device)
                attention_mask = attention_mask.to(nesy.args.task_device)
                y_pred = nesy.llm.predict_task(input_ids, attention_mask=attention_mask)
                #y_pred = [post_process_for_prompting(y) for y in y_pred]
                y_pred = [post_process_for_y(y) for y in y_pred]
                results = [
                    {
                        "knowledge": knowledge,
                        "x": x_batch[i],
                        "y_true": y_batch[i],
                        "y_pred": y_pred[i],
                        #"score": evaluater(y_pred[i], y_batch[i])
                        "score": evaluater(y_pred[i], y_batch[i], x_batch[i], knowledge)
                    }
                    for i in range(len(x_batch))
                ]

                for result in results:
                    log.writelines(f"{json.dumps(result, indent=4)}\n")
                    num_correct += result["score"]
                    num_test += 1
                    log.flush()

    accuracy = num_correct / num_test
    log.writelines(f"ICL accuracy on {name} samples: {accuracy} \n")
    log.flush()

def visualize_knowledge(args, epoch, seen_test_data, unseen_test_data, nesy, prompt_template, evaluater, name):

    all_tasks_ids = list(set([sample["sub_task_id"] for sample in seen_test_data]))

    #all_tasks_ids = random.sample(all_tasks_ids, 10)

    num_correct_symbolic = 0
    num_test_symbolic = 0

    all_groundtruth_knowledge = []
    all_encoded_knowledge = []

    with torch.no_grad():

        for task_id in tqdm(all_tasks_ids):

            # subtask_train_data = [data for data in train_data if data["sub_task_id"] == task_id]
            # subtask_test_data = [data for data in test_data if data["sub_task_id"] == task_id]

            subtask_data = [data for data in seen_test_data if data["sub_task_id"] == task_id]
            subtask_train_data = subtask_data#[:-1]
            subtask_valid_data = subtask_data[-1:]

            subtask_train_data_loader = DataLoader(subtask_train_data, batch_size=args.batch_size, shuffle=True)
            subtask_valid_data_loader = DataLoader(subtask_valid_data, batch_size=args.batch_size, shuffle=True)
            knowledge = subtask_valid_data[0]["knowledge"]
            all_groundtruth_knowledge.append(knowledge)

            knowledge_ids = nesy.llm.tokenizer(knowledge, return_tensors="pt").input_ids.to(nesy.args.encoder_device)
            encoded_mean, encoded_logvar = nesy.encode(knowledge_ids)
            all_encoded_knowledge.append(encoded_mean.to(torch.float16).cpu().numpy())
    
    combined_data = np.concatenate(all_encoded_knowledge, axis=0)
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_result = tsne.fit_transform(combined_data)

    # Create a DataFrame for Plotly
    df = pd.DataFrame(tsne_result, columns=['x', 'y'])
    df['knowledge'] = [knowledge for knowledge in all_groundtruth_knowledge]

    # 创建散点图,使用更深的蓝色
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        hover_data={'knowledge': True, 'x': False, 'y': False},
        color_discrete_sequence=['#0052CC'],  # 使用更深的蓝色 (#0052CC)
        template='plotly_white',
        width=1200,  # 设置固定宽度
        height=800  # 设置固定高度,保持正方形比例
    )

    # 保留坐标轴线但移除刻度和标签,设置浅灰色背景
    fig.update_layout(
        xaxis=dict(
            title=None,  # 移除x轴标签
            showticklabels=False,  # 移除刻度标签
            showgrid=True,
            gridcolor='white',  # 网格线设为白色
            zeroline=False,
            scaleanchor="y",  # 确保x和y轴使用相同的缩放
            scaleratio=1      # 保持1:1的比例
        ),
        yaxis=dict(
            title=None,  # 移除y轴标签
            showticklabels=False,  # 移除刻度标签
            showgrid=True,
            gridcolor='white',  # 网格线设为白色
            zeroline=False
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),  # 保留适当边距
        plot_bgcolor='#F5F5F5',  # 设置浅灰色背景
        paper_bgcolor='white'    # 图表外部背景保持白色
    )

    # 更新点的样式,增加不透明度
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,  # 增加不透明度
            line=dict(width=1, color='#E5E5E5')
        )
    )

    mkdir(f"{args.exp_dir}/epoch{epoch}")
    # 保存HTML和PDF格式,使用相同的尺寸设置
    fig.write_html(f"{args.exp_dir}/epoch{epoch}/latent.html")
    fig.write_image(f"{args.exp_dir}/epoch{epoch}/latent.pdf")

def main(args):

    if args.exp_name is None:
        current_time = datetime.now()
        args.exp_name = str(current_time)
    args.exp_dir = f"{args.meta_exp_dir}/{args.exp_name}"
    mkdir(args.exp_dir)
    
    if args.load_exp:
        if args.load_exp == "self":
            args.load_exp = args.exp_dir
        else:
            args.load_exp = f"{args.meta_exp_dir}/{args.load_exp}"
        with open(f"{args.load_exp}/args.json", "r") as f:
            loaded_args = json.load(f)
        for key in loaded_args:
            if key not in ["exp_dir", "load_exp", "load_epoch", "encoder_device", "decoder_device", "task_device", 
                           "flow_device", "noise_device", "task_finetune_step", "task_finetune_lr", "batch_size",
                           "zero_init", "dataset", "pretraining", "valid_epoch", "save_epoch", "task_model_name_or_path",
                           "model_name_or_path", "method", "test_sample_num", "dataset", "use_trainable_task_model", "indirect_finetune"]:
                args.__dict__[key] = loaded_args[key]
        args.load_nesy_ckpt = f"{args.load_exp}/epoch{args.load_epoch}/nesy_ckpt/"
        start_epoch = args.load_epoch
        file_mode = "w"
    else:
        # training from scratch
        args.load_nesy_ckpt = None
        start_epoch = 0
        file_mode = "w"

    if args.fuse_method in ["p-tuning", "delta", "lora"]:
        from transformers import AutoConfig
        task_model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        args.latent_size = args.num_soft_token * task_model_config.hidden_size
        print(f"latent_size now is: {args.latent_size}")

    args_dict = vars(args)
    output_file = f"{args.exp_dir}/args.json"
    with open(output_file, "w") as f:
        json.dump(args_dict, f, indent=4)
        f.flush()

    if args.observation_num is not None:
        args.test_sample_num = 6

    data = load_task_data(task=args.dataset, unseen_task_ratio=args.unseen_task_ratio, unseen_task_num=args.unseen_task_num,
                          test_sample_ratio=args.test_sample_ratio, test_sample_num=args.test_sample_num, 
                          num_words=args.num_words, num_pertask=args.num_pertask, task_fields=args.task_fields)
    
    if args.observation_num is not None:
        args.test_sample_num = args.observation_num

    args.task_id2knowledge, args.knowledge2task_id = create_task_data_lookup(data)
    prompt_template = data["prompt_template"]
    neural_evaluater = data["neural_evaluater"]
    symbolic_evaluater = data["symbolic_evaluater"]
    seen_train_data_loader = DataLoader(data["seen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    seen_test_data_loader = DataLoader(data["seen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)
    unseen_train_data_loader = DataLoader(data["unseen_tasks"]["train"], batch_size=args.batch_size, shuffle=True)
    unseen_test_data_loader = DataLoader(data["unseen_tasks"]["test"], batch_size=args.batch_size, shuffle=True)

    if args.pretraining:
        train_dataset, valid_dataset = load_pretrain_data_hf(pretrain_data_ratio=args.pretrain_data_ratio)
        if len(train_dataset) == 0:
            start_epoch = 1
            train_data_loader = None
            valid_data_loader = None
            print("evaluating without pretraining")
        else:
            train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
            print("pretraining")

    if args.prior == "gaussian":
        from vae import Nesy
    elif args.prior == "mog":
        from gmvae import Nesy
    elif args.prior == "gmg":
        from gmg import Nesy
    elif args.prior == "vaeflow":
        from vaeflow import Nesy
    else:
        raise Exception("undefined prior")
    
    if args.prior == "vaeflow":
        nesy = Nesy(args)#.to(torch.float32)
    else:
        nesy = Nesy(args).to(torch.float32)

    if args.method == "nesy":
        optimizer = torch.optim.Adam([
            {'params': nesy.llm.encoder.parameters(), 'lr': args.lr},
            {'params': nesy.encoder_mlp.parameters(), 'lr': args.lr},
            {'params': nesy.llm.decoder.parameters(), 'lr': args.lr},
            {'params': nesy.llm.task_model.parameters(), 'lr': args.lr},
            #{'params': nesy.decoder_mlp.parameters(), 'lr': args.lr},
            #{'params': nesy.flow_net.parameters(), 'lr': args.lr},
            #{'params': nesy.logZ, 'lr': args.lr}
                                    ], lr=args.lr)
        if args.prior == "vaeflow" and args.ebm_optim_method == "fce":
            optimizer_noise = torch.optim.Adam(nesy.noise_flow_net.parameters(), lr=args.lr*0.01)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
        train_log = open(f"{args.exp_dir}/train.log", file_mode)

        info = get_gpu_memory_usage()
        train_log.writelines(f"{info}\n")
        train_log.flush()

        train_data_loader = seen_train_data_loader if not args.pretraining else train_data_loader

        for epoch in range(start_epoch, args.num_epochs):

            if epoch % args.save_epoch == 0 and epoch > 0:
                
                nesy.save(f"{args.exp_dir}/epoch{epoch}/nesy_ckpt/")

            if epoch % args.valid_epoch == 0 and epoch > 0:

                mkdir(f"{args.exp_dir}/epoch{epoch}")

                neural2symbolic_test_log = open(f"{args.exp_dir}/epoch{epoch}/neural2symbolic.log", file_mode)
                test_neural2symbolic(args, epoch, data["seen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_test_log, name="seen task")
                test_neural2symbolic(args, epoch, data["unseen_tasks"]["test"], nesy, prompt_template, symbolic_evaluater, neural2symbolic_test_log, name="unseen task")

                symbolic2neural_test_log = open(f"{args.exp_dir}/epoch{epoch}/symbolic2neural.log", file_mode)
                test_symbolic2neural(args, epoch, seen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_test_log, name="seen task test")
                test_symbolic2neural(args, epoch, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, symbolic2neural_test_log, name="unseen task test")

            for i, batch in tqdm(enumerate(train_data_loader), desc=f"epoch {epoch}"):

                knowledge_batch = batch["knowledge"]
                x_batch = batch["input"]
                x_batch = [prompt_template.format(x) for x in x_batch]
                y_batch = batch["target"]

                optimizer.zero_grad()

                train_noise = False

                if args.prior == "gaussian":
                    reg_loss, recon_loss, task_loss = nesy.forward(knowledge_batch, x_batch, y_batch)
                    loss = args.reg_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss
                elif args.prior == "mog":
                    reg_loss, recon_loss, task_loss, entropy_loss = nesy.forward_batch(knowledge_batch, x_batch, y_batch)
                    loss = args.reg_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.task_loss_weight * task_loss #+ args.entropy_loss_weight * entropy_loss
                elif args.prior in ["gmg", "vaeflow"]:
                    
                    if nesy.args.ebm_optim_method == "fce":
                
                        kl_loss, recon_loss, task_loss, flow_loss, noise_loss, acc = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss
                    
                        train_noise = acc > args.threshold
                        train_log.writelines(f"acc={acc}\n")
                        train_log.writelines(f"train_noise={train_noise}\n")

                    elif nesy.args.ebm_optim_method in ["entropy", "kl"]:
                
                        kl_loss, recon_loss, task_loss, flow_loss, entropy = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * kl_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss - args.entropy_loss_weight * entropy
                    
                        train_log.writelines(f"entropy={entropy}\n")

                    else:
                        reg_loss, recon_loss, task_loss, flow_loss = nesy(knowledge_batch, x_batch, y_batch)
                        loss = args.kl_loss_weight * reg_loss + args.recon_loss_weight * recon_loss + args.flow_loss_weight * flow_loss #args.task_loss_weight * task_loss
                        
                if train_noise:

                    loss = noise_loss
                    loss.backward()
                    optimizer_noise.step()
                    if i % 10 == 0:
                        train_log.writelines(f"noise_loss={loss}\n")
                        train_log.flush()

                else:
                    loss.backward()
                    optimizer.step()

                    if i % 10 == 0:
                        train_log.writelines(f"epoch {epoch} step {i} \n")
                        if args.prior == "gaussian":
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, reg_loss={reg_loss}, task_loss={task_loss}\n")
                        elif args.prior == "mog":
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, reg_loss={reg_loss}, task_loss={task_loss}, entropy_loss={entropy_loss}\n")
                        elif args.prior in ["gmg", "vaeflow"]:
                            train_log.writelines(f"total_loss={loss}, recon_loss={recon_loss}, kl_loss={kl_loss}, flow_loss={flow_loss}\n")
                            train_log.writelines(f"task_loss={task_loss}\n")
                        train_log.flush()
                    
                if i % 100 == 0:
                    info = get_gpu_memory_usage()
                    train_log.writelines(f"{info}\n")
                    train_log.flush()

    elif args.method == "nesy_iterative":
        args.task_finetune_step /= 4
        log = open(f"{args.exp_dir}/refinement.log", "w")
        refinement_inference(args, seen_train_data_loader, seen_test_data_loader, nesy, prompt_template, neural_evaluater, log, name="seen task")
        refinement_inference(args, unseen_train_data_loader, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, log, name="unseen task")

    elif args.method in ["icl", "instruction_induction"]:
        icl_log = open(f"{args.exp_dir}/{args.method}.log", "w")
        icl_inference(args, seen_train_data_loader, seen_test_data_loader, nesy, prompt_template, neural_evaluater, icl_log, name="seen task")
        icl_inference(args, unseen_train_data_loader, unseen_test_data_loader, nesy, prompt_template, neural_evaluater, icl_log, name="unseen task")

    else:
        if args.method in ["prompting", "finetuning"]:
            symbolic_task_test_log = open(f"{args.exp_dir}/symbolic_task.log", "w")
            test_symbolic_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, 
                               prompt_template, symbolic_evaluater, symbolic_task_test_log, method=args.method)
        if args.method in ["prompting", "finetuning"]:
            neural_task_test_log = open(f"{args.exp_dir}/neural_task.log", "w")
            test_neural_task(args, seen_train_data_loader, seen_test_data_loader, unseen_test_data_loader, nesy, 
                             prompt_template, neural_evaluater, neural_task_test_log, method=args.method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="list_functions", help='name of dataset.')
    parser.add_argument('--meta_exp_dir', type=str, default="./exp", help='the directory to save all the experiment results.')
    parser.add_argument('--exp_name', type=str, default="debug", help='the name of the experiment.')
    parser.add_argument('--pretraining', action="store_true", default=False, help='Whether to pretrain the model.')

    parser.add_argument('--method', type=str, default="nesy", help='the method to train the model.')
    parser.add_argument('--prior', type=str, default="gaussian", help='the prior distribution of the model.')
    parser.add_argument('--nf', action="store_true", default=False, help='Whether to use the flow model.')
    # parser.add_argument('--fuse_method', type=str, default="delta", help='name of dataset.')
    parser.add_argument('--fuse_method', type=str, default="p-tuning", help='the method to fuse the task model and the prior model.')
    parser.add_argument('--use_instance_in_decoder', action="store_true", default=False, help='whether to use the instance in the decoder.')
    parser.add_argument('--use_knowledge_in_task', type=str, default="no", help='whether to use the instance in the decoder.')
    parser.add_argument('--use_trainable_task_model', action="store_true", default=False, help='whether to use the trainable task model.')
    parser.add_argument('--use_chat_template', action="store_true", default=False, help='whether to use the chat template.')
    parser.add_argument('--indirect_finetune', action="store_true", default=False, help='whether to use the chat template.')

    parser.add_argument('--ebm_optim_method', type=str, default="entropy", help='the method to optimize the energy-based model.')
    #parser.add_argument('--ebm_optim_method', type=str, default="nce", help='name of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='the beta parameter in the energy-based model.')
    parser.add_argument('--threshold', type=float, default=0.8, help='the threshold for the accuracy of the model.')

    parser.add_argument('--batch_size', type=int, default=4, help='the batch size.')
    parser.add_argument('--latent_size', type=int, default=1000, help='the dimension of the latent variable.')
    parser.add_argument('--selected_layers', type=int, default=2, help='the number of layers to be selected.')
    parser.add_argument('--num_latent_samples', type=int, default=2, help='the number of samples to be generated.')
    parser.add_argument('--num_peak', type=int, default=100, help='the number of peaks in the mixture of gaussians.')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate.')
    parser.add_argument('--episilon', type=float, default=1e-5, help='the episilon parameter in the energy-based model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs to train the model.')
    parser.add_argument('--valid_epoch', type=int, default=1, help='the number of epochs to validate the model.')
    parser.add_argument('--save_epoch', type=int, default=10, help='the number of epochs to save the model.')

    parser.add_argument('--task_finetune_step', type=int, default=30, help='the number of steps to finetune the task model.')
    parser.add_argument('--task_finetune_lr', type=float, default=1e-1, help='the learning rate to finetune the task model.')
    parser.add_argument('--zero_init', action="store_true", default=False, help='whether to initialize the task model parameters to zero.')

    parser.add_argument('--alignment_loss_weight', type=float, default=1, help='the weight of the alignment loss.')
    parser.add_argument('--task_loss_weight', type=float, default=1, help='the weight of the task loss.')
    parser.add_argument('--entropy_loss_weight', type=float, default=1e-5, help='the weight of the entropy loss.')
    parser.add_argument('--reg_loss_weight', type=float, default=0.01, help='the weight of the regularization loss.')
    parser.add_argument('--recon_loss_weight', type=float, default=1, help='the weight of the reconstruction loss.')
    parser.add_argument('--flow_loss_weight', type=float, default=10, help='the weight of the flow loss.')
    
    parser.add_argument('--max_token', type=int, default=50, help='max number of tokens to generate.')
    parser.add_argument('--num_soft_token', type=int, default=10, help='max number of tokens to generate.')
    
    #parser.add_argument('--load_exp', type=str, default="../exp_final/vae-domain-ls", help='name of dataset.')
    parser.add_argument('--load_exp', type=str, default=None, help='the path of the pretrained model.')
    parser.add_argument('--load_epoch', type=int, default=1, help='the epoch of the pretrained model.')
    parser.add_argument('--ignore_exist', action="store_true", default=False, help='whether to ignore the existing model.')
    parser.add_argument('--results_name', type=str, default=None, help='the name of the experiment.')
    parser.add_argument('--model_name_or_path', type=str, default="../llama-3-8b-instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct", help='Tasks for instructions generation')
    #parser.add_argument('--model_name_or_path', type=str, default="/cpfs/user/chenhao/pretrained_models/Qwen/Qwen2.5-7B", help='the path of the pretrained model.')
    parser.add_argument('--task_model_name_or_path', type=str, default=None, help='the path of the pretrained model.')
    parser.add_argument('--finetuned_model', type=str, default=None, help='the path of the finetuned model.')
    
    parser.add_argument('--cuda_devices', type=str, default="3,4,5", help='the devices to use')
    parser.add_argument('--encoder_device', type=int, default=0, help='the device to use')
    parser.add_argument('--decoder_device', type=int, default=1, help='the device to use')
    parser.add_argument('--task_device', type=int, default=2, help='the device to use')
    parser.add_argument('--flow_device', type=int, default=0, help='the device to use')
    parser.add_argument('--noise_device', type=int, default=4, help='device to use')
    parser.add_argument('--backward_device', type=int, default=0, help='device to use')
    
    parser.add_argument('--encoder_lora_r', type=int, default=16)
    parser.add_argument('--decoder_lora_r', type=int, default=1)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj", help='keywords must include in results')
        
    parser.add_argument('--num_words', type=int, default=32)
    parser.add_argument('--valid_ratio', type=float, default=0.01)
    parser.add_argument('--unseen_task_ratio', type=float, default=0.1)
    parser.add_argument('--unseen_task_num', type=int, default=None)
    parser.add_argument('--test_sample_ratio', type=float, default=None)
    parser.add_argument('--test_sample_num', type=int, default=5)
    parser.add_argument('--observation_num', type=int, default=None)
    parser.add_argument('--pretrain_data_ratio', type=float, default=1.0)
    parser.add_argument('--num_pertask', type=int, default=25)
    parser.add_argument('--task_fields', type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    main(args)