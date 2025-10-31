from json import load
import os
import random
import torch
import torch.nn as nn
from llm import WrappedLLM
from utils import mkdir
#import INN

class Nesy(nn.Module):
    
    def __init__(self, args):
        super(Nesy, self).__init__()
        self.args = args
        
        self.llm = WrappedLLM(self.args)
        self.hidden_size = self.llm.config.hidden_size
        self.latent_size = self.args.latent_size
        
        if args.method in ["nesy", "nesy_iterative"]:
            
            self.encoder_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.latent_size*2)
            ).to(self.args.encoder_device)#to(self.args.flow_device)
            
            # self.decoder_mlp = nn.Sequential(
            #     nn.Linear(self.latent_size, self.hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(self.hidden_size, self.hidden_size*self.args.num_soft_token),
            #     nn.Sigmoid()
            # ).to(self.args.decoder_device)#to(self.args.flow_device)

            if args.load_nesy_ckpt:
                self.load(args.load_nesy_ckpt)

            if args.nf:
                self.flow_net = INN.Sequential(
                    INN.Nonlinear(self.latent_size, method="RealNVP"),
                ).to(self.args.flow_device)

    def save(self, dir):
        mkdir(dir)
        torch.save(self.encoder_mlp.state_dict(), os.path.join(dir, "encoder_mlp.pth"))
        #torch.save(self.decoder_mlp.state_dict(), os.path.join(dir, "decoder_mlp.pth"))
        self.llm.save(dir)

    def load(self, dir):
        self.encoder_mlp.load_state_dict(torch.load(os.path.join(dir, "encoder_mlp.pth")))
        #self.decoder_mlp.load_state_dict(torch.load(os.path.join(dir, "decoder_mlp.pth")))
        self.llm.load(dir)

    def flow_forward(self, latent, return_all=False):
        #params = latent
        params, log_p, log_det_J = self.flow_net(latent.to(torch.float))
        # params = params.to(torch.bfloat16)
        # if return_all:
        #     return params, log_p, log_det_J
        # else:
        return params
    
    def flow_backward(self, params):
        #latent = params
        latent = self.flow_net.inverse(params.to(torch.float)).to(torch.float32)
        return latent

    def encode(self, knowledge_ids):
        outputs = self.llm.encode(knowledge_ids)
        last_hidden = outputs[:, -1, :]
        hidden = self.encoder_mlp(last_hidden)
        mean = hidden[:, :self.latent_size]
        log_var = hidden[:, self.latent_size:]
        return mean, log_var
    
    def compute_recon_loss(self, latent, labels, instance=None):
        #embedding = self.decoder_mlp(latent)#.to(self.args.decoder_device)
        embedding = latent
        if self.args.use_instance_in_decoder:
            instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            outputs = self.llm.decode(embedding, labels, instance_embedding)
        else:
            outputs = self.llm.decode(embedding, labels)
            # instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            # outputs = self.llm.decode(embedding, labels, instance_embedding)
        return outputs

    def predict_knowledge(self, context, sample_from_guassian=True, instance=None):
        
        if sample_from_guassian:
            sampled_latent = self.reparameterize(context, torch.ones_like(context)).to(self.args.decoder_device)
        else:
            sampled_latent = context
        embedding = sampled_latent#self.decoder_mlp(sampled_latent)
        if self.args.use_instance_in_decoder:
            instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            sampled_ids = self.llm.predict_knowledge(embedding, instance_embedding)
        else:
            sampled_ids = self.llm.predict_knowledge(embedding)
            # instance_embedding = self.llm.decoder_model.model.embed_tokens(instance)
            # sampled_ids = self.llm.predict_knowledge(embedding, instance_embedding)
        #text = [self.llm.tokenizer.decode(k) for k in sampled_ids.tolist()[0]]
        text = self.llm.tokenizer.decode(sampled_ids.tolist()[0], skip_special_tokens=True)
        text = text.split("<|endoftext|>")[0]
                
        return text

    def compute_kl_loss(self, mean, log_var):
        kl_loss = 0.5 * torch.mean(
            log_var.exp() + mean.pow(2) - 1 - log_var,
            dim=1
        )
        return kl_loss.mean()

    def compute_task_loss(self, latent, x_batch, y_batch, reduce=True):
        
        batch_size = len(x_batch)
        
        if self.args.fuse_method in ["delta", "lora"]:

            if reduce:
                task_loss = 0
            else:
                task_loss = []
         
            for i in range(batch_size):
                
                new_task_parameters = self.llm.allocate(latent[i])
                x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)
                y_id = self.llm.tokenizer(y_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)

                if reduce:
                    task_loss += self.llm.solve_task(x_id, y_id, new_task_parameters)
                else:
                    task_loss.append(self.llm.solve_task(x_id, y_id, new_task_parameters))

            if reduce:
                task_loss /= batch_size
            else:
                task_loss = torch.stack(task_loss, dim=0)
            
        elif self.args.fuse_method == "p-tuning":
            
            print("in compute_task_loss")
            print(f"x_batch: {random.choice(x_batch)}")
            print(f"y_batch: {random.choice(y_batch)}")

            x_id = self.llm.tokenizer(x_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            y_id = self.llm.tokenizer(y_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            
            if self.args.ebm_optim_method == "mc":
                x_id = x_id.repeat_interleave(self.args.num_latent_samples, dim=0)
                y_id = y_id.repeat_interleave(self.args.num_latent_samples, dim=0)
                latent = latent.reshape(batch_size*self.args.num_latent_samples, self.args.latent_size)
            else:
                latent = latent.reshape(batch_size, self.args.latent_size)
            
            task_loss = self.llm.solve_task(x_id, y_id, latent, reduce=reduce)
            
        return task_loss
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #eps = torch.randn((self.args.num_latent_samples, mean.shape[1])).to(mean.device).bfloat16()
        return mean + eps * std

    def forward(self, knowledge_batch, x_batch, y_batch):
        
        batch_size = len(knowledge_batch)

        knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
        mean, log_var = self.encode(knowledge_ids)

        reg_loss = self.compute_kl_loss(mean, log_var)

        sampled_latent = self.reparameterize(mean, log_var)

        sampled_latent = sampled_latent.to(self.args.decoder_device)
        knowledge_ids = knowledge_ids.to(self.args.decoder_device)
        
        if self.args.use_instance_in_decoder:
            instance = (x_batch, y_batch)
            instance_text = [f"input: {x}, target: {y}. This task is to:" for x, y in zip(*instance)]
            instance_ids = self.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.decoder_device)
        else:
            instance_ids = None
            # instance = (x_batch, y_batch)
            # instance_text = [f"This task is to:" for x, y in zip(*instance)]
            # instance_ids = self.llm.tokenizer(instance_text, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.decoder_device)
        recon_loss = self.compute_recon_loss(sampled_latent, knowledge_ids, instance_ids)

        if self.args.nf:
            sampled_latent = sampled_latent.to(self.args.flow_device)
            params = self.flow_forward(sampled_latent)
            params = params.to(self.args.task_device)
        else:
            params = sampled_latent.to(self.args.task_device)

        if self.args.use_knowledge_in_task.lower() in ["hard", "soft"]:
            if self.args.use_chat_template:
                x_batch = [self.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
            else:
                x_batch = [knowledge_batch[i] + x_batch[i] for i in range(len(x_batch))]
        else:
            if self.args.use_chat_template:
                x_batch = [self.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

        # new strategy
        # if self.args.use_knowledge_in_task.lower() in ["hard", "soft"]:
        #     if self.args.use_chat_template:
        #         chat_batch = [self.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, 
        #                                                            {"role": "user", "content": x_batch[i]},
        #                                                            {"role": "assistant", "content": y_batch[i]}],
        #                                                            tokenize=False) for i in range(len(x_batch))]
        #         if "llama" in self.args.model_name_or_path.lower():
        #             pattern = "<|start_header_id|>assistant"
        #         elif "qwen" in self.args.model_name_or_path.lower():
        #             pattern = "<|im_start|>assistant\n"
        #         else:
        #             raise ValueError(f"Unsupported model: {self.args.model_name_or_path}")
        #         x_batch = [chat.split(pattern)[0] for chat in chat_batch]
        #         y_batch = [pattern + chat.split(pattern)[1] for chat in chat_batch]
        #     else:
        #         x_batch = [knowledge_batch[i] + x_batch[i] for i in range(len(x_batch))]
        # else:
        #     if self.args.use_chat_template:
        #         x_batch = [self.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

        task_loss = self.compute_task_loss(params, x_batch, y_batch)

        #reg_loss = sampled_latent.norm(1, dim=1).mean() / self.args.latent_size

        recon_loss = recon_loss.to(self.args.backward_device)
        task_loss = task_loss.to(self.args.backward_device)
        reg_loss = reg_loss.to(self.args.backward_device)

        return reg_loss, recon_loss, task_loss

    def eval_task(self, knowledge_batch, x_batch, y_batch, evaluater, knowledge_groundtruth=None):
        
        batch_size = len(knowledge_batch)

        if knowledge_groundtruth is None:
            knowledge_groundtruth = knowledge_batch
        
        if self.args.use_knowledge_in_task.lower() in ["hard", "soft"]:
            if self.args.use_chat_template:
                x_batch = [self.llm.tokenizer.apply_chat_template([{"role": "system", "content": knowledge_batch[i]}, {"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]
            else:
                x_batch = [knowledge_batch[i] + x_batch[i] for i in range(len(x_batch))]
        else:
            if self.args.use_chat_template:
                x_batch = [self.llm.tokenizer.apply_chat_template([{"role": "user", "content": x_batch[i]}], tokenize=False) for i in range(len(x_batch))]

        if self.args.fuse_method in ["delta", "lora"]:
            
            results = []
            
            for i in range(batch_size):

                knowledge_ids = self.llm.tokenizer(knowledge_batch[i], add_special_tokens=True, return_tensors="pt").input_ids.to(self.args.encoder_device)#(self.args.device)
                mean, log_var = self.encode(knowledge_ids)

                latent = mean[0].to(self.args.task_device)
                
                new_task_parameters = self.llm.allocate(latent)
                
                x_id = self.llm.tokenizer(x_batch[i], return_tensors="pt", add_special_tokens=True).input_ids.to(self.args.task_device)
                
                y_pred = self.llm.predict_task(x_id, new_task_parameters)

                results.append({
                    "knowledge": knowledge_groundtruth[i],
                    "x": x_batch[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred,
                    "score": evaluater(y_pred, y_batch[i], x_batch[i], knowledge_groundtruth[i])
                    })

        elif self.args.fuse_method == "p-tuning":
            
            knowledge_ids = self.llm.tokenizer(knowledge_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.encoder_device)
            mean, log_var = self.encode(knowledge_ids)
            
            if self.args.nf:
                params = self.flow_forward(self.reparameterize(mean, log_var).to(self.args.flow_device)).to(self.args.task_device)
            else:
                params = self.reparameterize(mean, log_var).to(self.args.task_device)

            print("in eval_task")
            print(f"x_batch: {random.choice(x_batch)}")
            print(f"y_batch: {random.choice(y_batch)}")

            x_id = self.llm.tokenizer(x_batch, return_tensors="pt", add_special_tokens=True, padding="longest").input_ids.to(self.args.task_device)
            y_pred = self.llm.predict_task(x_id, params)
            
            results = [
                {
                    "knowledge": knowledge_groundtruth[i],
                    "x": x_batch[i],
                    "y_true": y_batch[i],
                    "y_pred": y_pred[i],
                    #"score": evaluater(y_pred[i], y_batch[i])
                    "score": evaluater(y_pred[i], y_batch[i], x_batch[i], knowledge_groundtruth[i])
                }
                for i in range(batch_size)
            ]
        return results
    
    def eval_knowledge(self, knowledge, predicted_knowledge, evaluater):

        result = {
            "groundtruth knowledge": knowledge,
            "predicted knowledge": predicted_knowledge,
            "score": evaluater(knowledge, predicted_knowledge)
            }

        return result