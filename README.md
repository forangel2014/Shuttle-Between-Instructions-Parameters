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

#### pretrain
```
bash run.sh --cuda_devices 0,1,2 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_final --exp_name vae-pretrain --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --pretrain_data_ratio 1 --num_pertask 27
```
```
bash run.sh --cuda_devices 0,6,7 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_p3new --exp_name vae-pretrain --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset p3 --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --pretrain_data_ratio 1 --num_pertask 25
```

#### in-domain
```
bash run.sh --cuda_devices 4,5,6 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_final --exp_name vae-domain-delta --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 10 --save_epoch 10 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining False --use_trainable_task_model True --use_knowledge_in_task soft --method nesy --num_pertask 27
```
```
bash run.sh --cuda_devices 0,1,2 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_p3new --exp_name vae-domain --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 10 --batch_size 4 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset p3 --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 10 --save_epoch 10 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining False --use_trainable_task_model True --use_knowledge_in_task soft --method nesy --num_pertask 25
```

### baselines
```
bash run.sh --cuda_devices 0,6,7 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_baseline --exp_name icl --method icl --test_sample_num 5
```

### induction
```
bash run.sh --cuda_devices 3,6,7 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_induction --exp_name vae-induction-1 --test_sample_num 1 --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --load_exp ../exp_final/vae-domain --load_epoch 10 --num_pertask 27
```
```
bash run.sh --cuda_devices 0,6,7 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_p3_induction --exp_name vae-induction-1 --test_sample_num 1 --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 4 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy --load_exp ../exp_p3/vae-domain --load_epoch 10 --num_pertask 25
```

### iterative
```
bash run.sh --cuda_devices 0,1,2 --model_name_or_path <your_pretrained_model_path> --meta_exp_dir ./exp_iterative --exp_name vae-iterative --recon_loss_weight 1 --reg_loss_weight 0.001 --task_loss_weight 1 --batch_size 8 --prior gaussian --unseen_task_ratio 0.1 --fuse_method p-tuning --num_soft_token 10 --dataset sni --encoder_lora_r 128 --decoder_lora_r 1 --valid_epoch 1 --save_epoch 1 --use_instance_in_decoder True --use_chat_template True --indirect_finetune True --pretraining True --use_trainable_task_model True --use_knowledge_in_task hard --method nesy_iterative --pretrain_data_ratio 1 --load_exp ../exp_final/vae-pretrain --load_epoch 1
```