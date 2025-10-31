### run

```
conda create -n nesyflow python==3.9
conda activate nesyflow
pip install -r requirements.txt
git clone https://github.com/ELIFE-ASU/INNLab
cd INNLab/
python setup.py install
cd ..
```

#### SHIP-pretrain
```
bash run.sh --cuda_devices 0,1 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_qs_sni --exp_name qwen-pretrain-ptuning --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method delta --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 128 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model False --use_knowledge_in_task hard --method nesy --pretrain_data_ratio 1 --num_pertask 27 --task_device 0
```

#### SHIP-domain
```
bash run.sh --cuda_devices 0,1 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_qs_sni --exp_name qwen-domain-ptuning --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --batch_size 4 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 128 --valid_epoch 10 --save_epoch 10 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining False --use_trainable_task_model False --use_knowledge_in_task hard --method nesy --pretrain_data_ratio 1 --num_pertask 27 --task_device 0
```

#### Inductive Reasoning
```
bash run.sh --cuda_devices 0,1 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_qs_sni --exp_name qwen-pretrain-iterative --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --batch_size 4 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 128 --valid_epoch 10 --save_epoch 10 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining False --use_trainable_task_model False --use_knowledge_in_task hard --method nesy_iterative --pretrain_data_ratio 1 --num_pertask 27 --task_device 0 --load_exp ./qwen-pretrain-ptuning --load_epoch 1
```