from multiprocessing import reduction
import os
import re
import random
import torch
import copy
import json
import torch.nn as nn
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from peft import AutoPeftModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaTokenizerFast
from utils import mkdir

class WrappedLLM(nn.Module):
    
    def __init__(self, args):
        super(WrappedLLM, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.dtype = torch.float32

        if args.task_model_name_or_path is None:
            args.task_model_name_or_path = args.model_name_or_path

        self.task_model_base = AutoModelForCausalLM.from_pretrained(args.task_model_name_or_path,
                                                        device_map=args.task_device,#"auto",
                                                        torch_dtype=self.dtype, 
                                                        trust_remote_code=True,
                                                        #torch_dtype=torch.float16, 
                                                        #load_in_8bit=True
                                                        )
        
        if args.use_trainable_task_model:
            self.task_config = LoraConfig(
                r=args.decoder_lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(","),
                fan_in_fan_out=False,
                lora_dropout=0.05,
                inference_mode=False,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.task_model = get_peft_model(self.task_model_base, self.task_config)
            self.task_model.print_trainable_parameters()
        else:
            self.task_model = self.task_model_base
            for params in self.task_model.parameters():
                params.requires_grad = False

        if "llama" in args.model_name_or_path.lower():
            #self.tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)
            self.tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)
            #self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)
            self.tokenizer.pad_token_id = 0
        elif "qwen" in args.model_name_or_path.lower():
            base_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)
            base_tokenizer.pad_token_id = 151656
            class WrappedTokenizer:
                def __init__(self, base_tokenizer):
                    self.base_tokenizer = base_tokenizer
                    for attr in dir(base_tokenizer):
                        if not attr.startswith('__'):
                            setattr(self, attr, getattr(base_tokenizer, attr))
                            
                def __call__(self, *args, **kwargs):
                    args = list(args)
                    if type(args[0]) == str:
                        args[0] = args[0] + "<|endoftext|>"
                    elif type(args[0]) == list:
                        args[0] = [x + "<|endoftext|>" for x in args[0]]
                    else:
                        raise ValueError(f"Unsupported input type: {type(args[0])}")
                    return self.base_tokenizer(*args, **kwargs)
            self.tokenizer = WrappedTokenizer(base_tokenizer)
            #self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side='left', add_bos_token=False, add_eos_token=True)

        if args.method in ["nesy", "nesy_iterative", "finetuning"]:

            self.encoder_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                            device_map=args.encoder_device,#"auto",
                                                            torch_dtype=self.dtype, 
                                                            trust_remote_code=True,
                                                            #torch_dtype=torch.float16, 
                                                            #load_in_4bit=True
                                                            )
            self.encoder_config = LoraConfig(
                r=args.encoder_lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(","),
                fan_in_fan_out=False,
                lora_dropout=0.05,
                inference_mode=False,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.decoder_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                            device_map=args.decoder_device,#"auto",
                                                            torch_dtype=self.dtype,
                                                            trust_remote_code=True,
                                                            #torch_dtype=torch.float16, 
                                                            #load_in_4bit=True
                                                            )
            self.decoder_config = LoraConfig(
                r=args.decoder_lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(","),
                fan_in_fan_out=False,
                lora_dropout=0.05,
                inference_mode=False,
                bias="none",
                task_type="CAUSAL_LM",
            )

            if args.load_nesy_ckpt:
                #self.load(args.load_nesy_ckpt)
                pass
            else:
                # if args.use_trainable_task_model:
                #     self.task_model = get_peft_model(self.task_model, self.task_config)
                #     self.task_model.print_trainable_parameters()
                self.encoder = get_peft_model(self.encoder_model.model, self.encoder_config)
                self.encoder.print_trainable_parameters()
                self.decoder = get_peft_model(self.decoder_model, self.decoder_config)
                self.decoder.print_trainable_parameters()
                self.param_info = self.specify_parameter(n=args.latent_size)
        
        elif args.method == "finetuning":
            self.param_info = self.specify_parameter(n=args.latent_size)

        elif args.method in ["nesy_visualize"]:
            self.encoder_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map=args.encoder_device,#"auto",
                                                torch_dtype=self.dtype, 
                                                trust_remote_code=True,
                                                #torch_dtype=torch.float16, 
                                                #load_in_4bit=True
                                                )
            
            self.encoder_config = LoraConfig(
                r=args.encoder_lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(","),
                fan_in_fan_out=False,
                lora_dropout=0.05,
                inference_mode=False,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder_model.model, self.encoder_config)
            self.encoder.print_trainable_parameters()

    def save(self, dir):
        if self.args.use_trainable_task_model:
            self.task_model.save_pretrained(os.path.join(dir, "task_model_lora"))
        self.encoder.save_pretrained(os.path.join(dir, "encoder_lora"))
        self.decoder.save_pretrained(os.path.join(dir, "decoder_lora"))
        json.dump(self.param_info, open(os.path.join(dir, "params_info.json"), "w"))

    def load(self, dir):
        if self.args.use_trainable_task_model:
            self.task_model = PeftModel.from_pretrained(self.task_model_base, os.path.join(dir, "task_model_lora")).to(self.args.task_device)
        self.encoder = PeftModel.from_pretrained(self.encoder_model.model, os.path.join(dir, "encoder_lora")).to(self.args.encoder_device)
        self.decoder = PeftModel.from_pretrained(self.decoder_model, os.path.join(dir, "decoder_lora")).to(self.args.decoder_device)
        self.param_info = json.load(open(os.path.join(dir, "params_info.json"), "r"))

    def specify_parameter(self, n):
        
        if self.args.fuse_method == "delta":
            
            param_counts = {}

            num_layers = self.task_model.model.config.num_hidden_layers
        
            selected_layer_id = [f".{num_layers-1-i}." for i in range(self.args.selected_layers)]
            for name, params in dict(self.task_model.named_parameters()).items():
                if params.dtype == self.dtype and "layers" in name and "_proj" in name:
                    if any([id_ in name for id_ in selected_layer_id]):
                        param_counts[name] = params.view(-1).shape[0]

            param_count_sum = sum(param_counts.values())
            param_allocation = {}
            for name, count in param_counts.items():
                param_allocation[name] = int(n * count / param_count_sum)

            param_info = []
            for name, specified_param_num in param_counts.items():
                params = dict(self.task_model.named_parameters())[name]
                sampled_param_num = param_allocation[name]
                weights = params.view(-1)
                indices = random.sample(range(weights.size(0)), sampled_param_num)
                #selected_weights = weights[indices].detach()
                indices = [[indice % params.shape[0] for indice in indices], [indice // params.shape[0] for indice in indices]]

                param_info.append((name, indices, sampled_param_num))#weights.shape, selected_weights))
        
        elif self.args.fuse_method == "lora":

            # 构造 LoRA 外部参数块，保证总参数量精确等于 n
            # 规则：仅在选定的若干层的线性投影（如 q/k/v/o/gate/up/down）上添加 LoRA 分支
            # 一个 LoRA 块的参数量 = rank * (|in_indices| + |out_indices|)
            # 我们优先以完整通道集构造 rank=1 的块，若有余数，再用部分通道的块凑齐精确的 n

            # 收集候选权重（线性层）
            num_layers = self.task_model.model.config.num_hidden_layers
            selected_layer_id = [f".{num_layers-1-i}." for i in range(self.args.selected_layers)]
            target_modules = [m.strip() for m in self.args.target_modules.split(",")] if hasattr(self.args, "target_modules") else [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]

            candidates = []  # (name, in_features, out_features)
            for name, params in dict(self.task_model.named_parameters()).items():
                if params.dtype != self.dtype:
                    continue
                if "layers" not in name:
                    continue
                if ".weight" not in name:
                    continue
                if not any([id_ in name for id_ in selected_layer_id]):
                    continue
                if not any([((tm + ".weight") in name or (tm + ".base_layer.weight") in name) for tm in target_modules]):
                    continue
                # 线性层权重形状为 [out_features, in_features]
                if len(params.shape) != 2:
                    continue
                out_features, in_features = params.shape[0], params.shape[1]
                candidates.append((name, in_features, out_features))

            if len(candidates) == 0:
                raise ValueError("LoRA 融合未找到可用的候选线性层，请检查 target_modules 与 selected_layers 设置。")

            # 依据 in+out 排序（优先在更大的矩阵上放置 rank=1 的块）
            candidates.sort(key=lambda x: (x[1] + x[2]), reverse=True)

            remaining = int(n)
            if remaining <= 0:
                return []
            if remaining == 1:
                raise ValueError("LoRA 外部参数个数 n=1 无法构造有效的 LoRA 块（至少需要 2 个参数）。")

            param_info = []  # 列表，每个元素是一个 LoRA 块的描述 dict（JSON 可序列化）
            min_unit = min(ci + co for _, ci, co in candidates)

            # 贪心放置 rank=1 的完整通道块
            placed_blocks = []  # 记录已经放置的完整块，便于处理 leftover=1 的情况
            while remaining >= min_unit:
                progressed = False
                for name, in_features, out_features in candidates:
                    unit = in_features + out_features
                    if remaining >= unit:
                        in_indices = list(range(in_features))
                        out_indices = list(range(out_features))
                        block = {
                            "name": name,
                            "type": "lora",
                            "rank": 1,
                            "alpha": int(getattr(self.args, "lora_alpha", 16)),
                            "in_indices": in_indices,
                            "out_indices": out_indices,
                            "in_features": in_features,
                            "out_features": out_features,
                            "num_params": unit,
                        }
                        param_info.append(block)
                        placed_blocks.append((name, in_features, out_features))
                        remaining -= unit
                        progressed = True
                        if remaining < min_unit:
                            break
                if not progressed:
                    break

            # 处理剩余参数
            if remaining > 0:
                # 优先尝试整倍数分配到某个矩阵的多 rank
                divisible_candidate = None
                for name, in_features, out_features in candidates:
                    unit = in_features + out_features
                    if remaining % unit == 0:
                        r = remaining // unit
                        if r > 0:
                            in_indices = list(range(in_features))
                            out_indices = list(range(out_features))
                            block = {
                                "name": name,
                                "type": "lora",
                                "rank": int(r),
                                "alpha": int(getattr(self.args, "lora_alpha", 16)),
                                "in_indices": in_indices,
                                "out_indices": out_indices,
                                "in_features": in_features,
                                "out_features": out_features,
                                "num_params": int(r) * unit,
                            }
                            param_info.append(block)
                            remaining = 0
                            divisible_candidate = True
                            break
                if not divisible_candidate and remaining > 0:
                    # 避免出现 remaining==1 的不可行情况
                    if remaining == 1:
                        if len(placed_blocks) == 0:
                            raise ValueError("LoRA 参数剩余为 1 且无可调整的已放置块，无法精确凑齐 n。请增大 n。")
                        # 回退一个完整块并用部分通道 +1 重新放置
                        back_name, back_in, back_out = placed_blocks.pop()
                        # 从 param_info 中移除该块
                        for i in range(len(param_info)-1, -1, -1):
                            if param_info[i]["name"] == back_name and param_info[i]["rank"] == 1 and \
                               len(param_info[i]["in_indices"]) == back_in and len(param_info[i]["out_indices"]) == back_out:
                                param_info.pop(i)
                                break
                        remaining += (back_in + back_out)
                        # 现在 remaining >= 2，一次性用部分通道凑满
                    # 构造一个 rank=1 的部分通道块，使 in_sub + out_sub == remaining
                    # 选择一个容量最大的候选
                    sel_name, sel_in, sel_out = candidates[0]
                    # 计算子通道数，确保两者都 > 0
                    in_sub = max(1, min(sel_in, remaining - 1))
                    out_sub = remaining - in_sub
                    if out_sub == 0:
                        out_sub = 1
                        if in_sub > 1:
                            in_sub -= 1
                    if out_sub > sel_out:
                        out_sub = sel_out
                        in_sub = remaining - out_sub
                    if in_sub <= 0 or out_sub <= 0 or in_sub > sel_in or out_sub > sel_out:
                        raise ValueError("无法为 LoRA 精确划分部分通道，请调整 n 或 target_modules/selected_layers。")
                    # 采样通道索引（随机/可重复）
                    in_indices = random.sample(range(sel_in), in_sub) if in_sub < sel_in else list(range(sel_in))
                    out_indices = random.sample(range(sel_out), out_sub) if out_sub < sel_out else list(range(sel_out))
                    block = {
                        "name": sel_name,
                        "type": "lora",
                        "rank": 1,
                        "alpha": int(getattr(self.args, "lora_alpha", 16)),
                        "in_indices": in_indices,
                        "out_indices": out_indices,
                        "in_features": sel_in,
                        "out_features": sel_out,
                        "num_params": int(1 * (in_sub + out_sub)),
                    }
                    param_info.append(block)
                    remaining = 0

        elif self.args.fuse_method == "p-tuning":
            param_info = {}

        else:
            raise ValueError(f"Unsupported fuse method: {self.args.fuse_method}")
        
        return param_info
    
    def allocate(self, delta_params):
        
        used_idx = 0
        new_task_parameters = {}
        
        if self.args.fuse_method == "delta":
            for i in range(len(self.param_info)):
                name, indices, sampled_param_num = self.param_info[i]
                new_weight = delta_params[used_idx:used_idx+sampled_param_num]
                used_idx += sampled_param_num
                new_task_parameters[name] = (indices, new_weight)
        elif self.args.fuse_method == "lora":
            block_id_counters = {}
            for block in self.param_info:
                name = block["name"]
                rank = int(block["rank"])
                alpha = int(block["alpha"])
                in_indices = block["in_indices"]
                out_indices = block["out_indices"]
                in_count = len(in_indices)
                out_count = len(out_indices)
                num_params = rank * (in_count + out_count)

                a_flat = delta_params[used_idx:used_idx + rank * in_count]
                used_idx += rank * in_count
                b_flat = delta_params[used_idx:used_idx + rank * out_count]
                used_idx += rank * out_count

                A = a_flat.view(in_count, rank) / 100
                B = b_flat.view(out_count, rank) / 100

                block_id_counters[name] = block_id_counters.get(name, 0) + 1
                key = f"{name}::lora::{block_id_counters[name]}"
                new_task_parameters[key] = {
                    "type": "lora",
                    "rank": rank,
                    "alpha": alpha,
                    "in_indices": in_indices,
                    "out_indices": out_indices,
                    "A": A,
                    "B": B,
                }
        else:
            raise ValueError(f"Unsupported fuse method in allocate: {self.args.fuse_method}")

        return new_task_parameters
        
    def reset(self):
        
        for i in range(len(self.param_info)):

            name, idx, weight = self.param_info[i]
            dict(self.task_model.named_parameters())[name].view(-1)[idx].copy_(weight)

    def encode(self, inputs):
        if inputs.dim() == 2:
            attention_mask = inputs != self.tokenizer.pad_token_id
            outputs = self.encoder(inputs, attention_mask=attention_mask)
        else:
            outputs = self.encoder(inputs_embeds=inputs)

        return outputs[0]#.float()

    def decode(self, embedding, labels, instance_embedding=None):
        attention_mask = labels != self.tokenizer.pad_token_id
        inputs_embeds = self.decoder_model.model.embed_tokens(labels)#.repeat(embedding.shape[0], 1, 1)
        #labels = labels.repeat(embedding.shape[0], 1)
        # if embedding.dim() == 2:
        #     embedding = embedding.unsqueeze(1)
        soft_token_embedding = embedding.view(embedding.shape[0], self.args.num_soft_token, self.config.hidden_size)

        if self.args.use_instance_in_decoder:
            soft_token_embedding = torch.cat((soft_token_embedding, instance_embedding), dim=1)
        # else:
        #     soft_token_embedding = torch.cat((soft_token_embedding, instance_embedding), dim=1)

        total_embeds = torch.cat((soft_token_embedding, inputs_embeds), dim=1)
        pad_tokens = torch.full_like(soft_token_embedding[:, :, 0], self.tokenizer.pad_token_id, dtype=torch.int)
        total_labels = torch.cat((pad_tokens, labels), dim=1)
        total_labels[total_labels==self.tokenizer.pad_token_id] = -100
        pad_attention = torch.full_like(soft_token_embedding[:, :, 0], 1, dtype=torch.int)
        total_attention = torch.cat((pad_attention, attention_mask), dim=1)
        outputs = self.decoder(inputs_embeds=total_embeds, attention_mask=total_attention, labels=total_labels)

        return outputs[0]#.float()

    def solve_task(self, x_id, y_id, new_task_parameters, reduce=True):
                
        if self.args.fuse_method in ["delta", "lora"]:
        
            input_ids = torch.cat((x_id, y_id), dim=1)
            pad_tokens = torch.full_like(x_id, self.tokenizer.pad_token_id, dtype=torch.int)
            labels = torch.cat((pad_tokens, y_id), dim=1)
            labels[labels==self.tokenizer.pad_token_id] = -100

            outputs = self.task_model(input_ids=[input_ids, new_task_parameters], labels=labels)

        elif self.args.fuse_method == "p-tuning":

            batch_size = new_task_parameters.shape[0]

            input_ids = torch.cat((x_id, y_id), dim=1)
            if self.args.use_trainable_task_model:
                inputs_embeds = self.task_model.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.task_model.model.embed_tokens(input_ids)

            if self.args.ebm_optim_method == "mc":
                soft_token_embedding = new_task_parameters.view(batch_size*self.args.num_latent_samples, self.args.num_soft_token, self.config.hidden_size)
            else:
                soft_token_embedding = new_task_parameters.view(batch_size, self.args.num_soft_token, self.config.hidden_size)

            attention_mask = input_ids != self.tokenizer.pad_token_id
            pad_attention = torch.full_like(soft_token_embedding[:, :, 0], 1, dtype=torch.int)
            total_attention = torch.cat((pad_attention, attention_mask), dim=1)
            
            total_embeds = torch.cat((soft_token_embedding, inputs_embeds), dim=1)
            pad_tokens_soft = torch.full_like(soft_token_embedding[:, :, 0], self.tokenizer.pad_token_id, dtype=torch.int)
            pad_tokens_x = torch.full_like(x_id, self.tokenizer.pad_token_id, dtype=torch.int)
            total_labels = torch.cat((pad_tokens_soft, pad_tokens_x, y_id), dim=1)
            total_labels[total_labels==self.tokenizer.pad_token_id] = -100

            outputs = self.task_model(inputs_embeds=total_embeds, attention_mask=[total_attention, reduce], labels=total_labels)

        return outputs[0]#.float()

    def predict_task(self, x_id, new_task_parameters=None, sample=False):
        
        if self.args.fuse_method in ["delta", "lora"]:
            
            if new_task_parameters is not None:
                inputs = [x_id, new_task_parameters]
            else:
                inputs = x_id
                
            response = self.task_model.generate(inputs=inputs, 
                                    max_new_tokens=self.args.max_token, 
                                    early_stopping=True,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    temperature=0.0,
                                    do_sample=False,
                                    # stopping_criteria=stopping_criteria
                                    )

            if response.shape[0] == 1:
                decoded_tokens = response[0][x_id.shape[1]:]
                text = self.tokenizer.decode(decoded_tokens, skip_special_tokens=True).replace(self.tokenizer.pad_token, "")
            else:
                decoded_tokens = [response[i][x_id.shape[1]:] for i in range(response.shape[0])]
                text = [self.tokenizer.decode(tokens, skip_special_tokens=True).replace(self.tokenizer.pad_token, "") for tokens in decoded_tokens]

            if type(text) == list:
                text = [t.split("<|endoftext|>")[0] for t in text]
            else:
                text = text.split("<|endoftext|>")[0]

        elif self.args.fuse_method == "p-tuning":
            
            batch_size = x_id.size(0)
            if new_task_parameters is not None:
                soft_token_embedding = new_task_parameters.view(batch_size, self.args.num_soft_token, self.config.hidden_size)
                if self.args.use_trainable_task_model:
                    inputs_embeds = self.task_model.model.model.embed_tokens(x_id)
                else:
                    inputs_embeds = self.task_model.model.embed_tokens(x_id)
                total_embeds = torch.cat((soft_token_embedding, inputs_embeds), dim=1)

            else:
                inputs_embeds = self.task_model.model.embed_tokens(x_id)
                total_embeds = inputs_embeds

            if new_task_parameters is not None:
                attention_mask = x_id != self.tokenizer.pad_token_id
                pad_attention = torch.full_like(soft_token_embedding[:, :, 0], 1, dtype=torch.int)
                total_attention = torch.cat((pad_attention, attention_mask), dim=1)

            else:
                attention_mask = x_id != self.tokenizer.pad_token_id
                total_attention = attention_mask

            if sample:
                response = self.task_model.generate(inputs=x_id,
                                        attention_mask=total_attention,
                                        max_new_tokens=self.args.max_token, 
                                        early_stopping=True,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        temperature=1.0,
                                        do_sample=True,
                                        # stopping_criteria=stopping_criteria
                                        )[:, x_id.shape[1]:]
        
            else:
                if self.args.method == "icl":
                    response = self.task_model.generate(inputs=x_id,
                                        max_new_tokens=self.args.max_token, 
                                        early_stopping=True,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        #temperature=0.0,
                                        #do_sample=False,
                                        # stopping_criteria=stopping_criteria
                                        )[:, x_id.shape[1]:]
                else:
                    response = self.task_model.generate(inputs_embeds=total_embeds,
                                        attention_mask=total_attention,
                                        max_new_tokens=self.args.max_token, 
                                        early_stopping=True,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        #temperature=0.0,
                                        #do_sample=False,
                                        # stopping_criteria=stopping_criteria
                                        )
            
            text = [self.tokenizer.decode(response[i], skip_special_tokens=True).split("<|endoftext|>")[0].replace(self.tokenizer.pad_token, "") for i in range(batch_size)]
        
        return text

    def predict_knowledge(self, embedding, instance_embedding=None):
        
        # if embedding.dim() == 2:
        #     embedding = embedding.unsqueeze(1)
        embedding = embedding.view(embedding.shape[0], self.args.num_soft_token, self.config.hidden_size)
        
        if instance_embedding is not None:
            embedding = torch.cat((embedding, instance_embedding), dim=1)
        
        #embedding = embedding.float32()
        
        response = self.decoder_model.generate(inputs_embeds=embedding, 
                                max_new_tokens=self.args.max_token, 
                                early_stopping=True,
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                #temperature=0.0,
                                #do_sample=False,
                                # stopping_criteria=stopping_criteria
                                )


        return response