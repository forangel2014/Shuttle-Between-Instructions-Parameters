#!/bin/bash

# 在开始处理参数之前保存原始命令行参数
ORIGINAL_ARGS="$@"

pip install plotly

cd patch
bash install.sh
cd ..

meta_exp_dir="exp"
exp_name="debug"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda_devices) cuda_devices="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --model_name_or_path) model_name_or_path="$2"; shift ;;
        --task_model_name_or_path) task_model_name_or_path="$2"; shift ;;
        --meta_exp_dir) meta_exp_dir="$2"; shift ;;
        --exp_name) exp_name="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --pretraining) pretraining="$2"; shift ;;
        --use_instance_in_decoder) use_instance_in_decoder="$2"; shift ;;
        --use_knowledge_in_task) use_knowledge_in_task="$2"; shift ;;
        --use_trainable_task_model) use_trainable_task_model="$2"; shift ;;
        --use_chat_template) use_chat_template="$2"; shift ;;
        --indirect_finetune) indirect_finetune="$2"; shift ;;
        --method) method="$2"; shift ;;
        --num_peak) num_peak="$2"; shift ;;
        --valid_epoch) valid_epoch="$2"; shift ;;
        --save_epoch) save_epoch="$2"; shift ;;
        --fuse_method) fuse_method="$2"; shift ;;
        --nf) nf="$2"; shift ;;
        --ebm_optim_method) ebm_optim_method="$2"; shift ;;
        --prior) prior="$2"; shift ;;
        --alignment_loss_weight) alignment_loss_weight="$2"; shift ;;
        --task_loss_weight) task_loss_weight="$2"; shift ;;
        --entropy_loss_weight) entropy_loss_weight="$2"; shift ;;
        --reg_loss_weight) reg_loss_weight="$2"; shift ;;
        --recon_loss_weight) recon_loss_weight="$2"; shift ;;
        --flow_loss_weight) flow_loss_weight="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_soft_token) num_soft_token="$2"; shift ;;
        --beta) beta="$2"; shift ;;
        --selected_layers) selected_layers="$2"; shift ;;
        --latent_size) latent_size="$2"; shift ;;
        --encoder_device) encoder_device="$2"; shift ;;
        --decoder_device) decoder_device="$2"; shift ;;
        --task_device) task_device="$2"; shift ;;
        --flow_device) flow_device="$2"; shift ;;
        --backward_device) backward_device="$2"; shift ;;
        --num_latent_samples) num_latent_samples="$2"; shift ;;
        --load_exp) load_exp="$2"; shift ;;
        --pretrain_data_ratio) pretrain_data_ratio="$2"; shift ;;
        --num_pertask) num_pertask="$2"; shift ;;
        --unseen_task_ratio) unseen_task_ratio="$2"; shift ;;
        --test_sample_num) test_sample_num="$2"; shift ;;
        --observation_num) observation_num="$2"; shift ;;
        --test_sample_ratio) test_sample_ratio="$2"; shift ;;
        --load_epoch) load_epoch="$2"; shift ;;
        --encoder_lora_r) encoder_lora_r="$2"; shift ;;
        --decoder_lora_r) decoder_lora_r="$2"; shift ;;
        --lora_alpha) lora_alpha="$2"; shift ;;
        --target_modules) target_modules="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python_cmd="python main.py"
args=""

if [ -n "$model_name_or_path" ]; then
    args="$args --model_name_or_path $model_name_or_path"
fi

if [ -n "$task_model_name_or_path" ]; then
    args="$args --task_model_name_or_path $task_model_name_or_path"
fi

if [ -n "$cuda_devices" ]; then
    args="$args --cuda_devices $cuda_devices"
fi

if [ -n "$dataset" ]; then
    args="$args --dataset $dataset"
fi

if [ -n "$meta_exp_dir" ]; then
    args="$args --meta_exp_dir $meta_exp_dir"
fi

if [ "$pretraining" = "True" ] || [ "$pretraining" = "true" ]; then
    args="$args --pretraining"
fi

if [ "$use_instance_in_decoder" = "True" ] || [ "$use_instance_in_decoder" = "true" ]; then
    args="$args --use_instance_in_decoder"
fi

if [ "$use_trainable_task_model" = "True" ] || [ "$use_trainable_task_model" = "true" ]; then
    args="$args --use_trainable_task_model"
fi

if [ "$use_chat_template" = "True" ] || [ "$use_chat_template" = "true" ]; then
    args="$args --use_chat_template"
fi

if [ "$indirect_finetune" = "True" ] || [ "$indirect_finetune" = "true" ]; then
    args="$args --indirect_finetune"
fi

if [ -n "$use_knowledge_in_task" ]; then
    args="$args --use_knowledge_in_task $use_knowledge_in_task"
fi

if [ -n "$num_peak" ]; then
    args="$args --num_peak $num_peak"
fi

if [ -n "$exp_name" ]; then
    args="$args --exp_name $exp_name"
fi

if [ -n "$method" ]; then
    args="$args --method $method"
fi

if [ -n "$ebm_optim_method" ]; then
    args="$args --ebm_optim_method $ebm_optim_method"
fi

if [ -n "$test_sample_num" ]; then
    args="$args --test_sample_num $test_sample_num"
fi

if [ -n "$lr" ]; then
    args="$args --lr $lr"
fi

if [ -n "$valid_epoch" ]; then
    args="$args --valid_epoch $valid_epoch"
fi

if [ -n "$save_epoch" ]; then
    args="$args --save_epoch $save_epoch"
fi

if [ -n "$prior" ]; then
    args="$args --prior $prior"
fi

if [ "$nf" = "True" ] || [ "$nf" = "true" ]; then
    args="$args --nf"
fi

if [ -n "$alignment_loss_weight" ]; then
    args="$args --alignment_loss_weight $alignment_loss_weight"
fi

if [ -n "$reg_loss_weight" ]; then
    args="$args --reg_loss_weight $reg_loss_weight"
fi

if [ -n "$task_loss_weight" ]; then
    args="$args --task_loss_weight $task_loss_weight"
fi

if [ -n "$entropy_loss_weight" ]; then
    args="$args --entropy_loss_weight $entropy_loss_weight"
fi

if [ -n "$recon_loss_weight" ]; then
    args="$args --recon_loss_weight $recon_loss_weight"
fi

if [ -n "$flow_loss_weight" ]; then
    args="$args --flow_loss_weight $flow_loss_weight"
fi

if [ -n "$batch_size" ]; then
    args="$args --batch_size $batch_size"
fi

if [ -n "$num_soft_token" ]; then
    args="$args --num_soft_token $num_soft_token"
fi

if [ -n "$beta" ]; then
    args="$args --beta $beta"
fi

if [ -n "$num_latent_samples" ]; then
    args="$args --num_latent_samples $num_latent_samples"
fi

if [ -n "$selected_layers" ]; then
    args="$args --selected_layers $selected_layers"
fi

if [ -n "$latent_size" ]; then
    args="$args --latent_size $latent_size"
fi

if [ -n "$encoder_device" ]; then
    args="$args --encoder_device $encoder_device"
fi

if [ -n "$decoder_device" ]; then
    args="$args --decoder_device $decoder_device"
fi

if [ -n "$task_device" ]; then
    args="$args --task_device $task_device"
fi

if [ -n "$backward_device" ]; then
    args="$args --backward_device $backward_device"
fi

if [ -n "$flow_device" ]; then
    args="$args --flow_device $flow_device"
fi

if [ -n "$unseen_task_ratio" ]; then
    args="$args --unseen_task_ratio $unseen_task_ratio"
fi

if [ -n "$test_sample_ratio" ]; then
    args="$args --test_sample_ratio $test_sample_ratio"
fi

if [ -n "$observation_num" ]; then
    args="$args --observation_num $observation_num"
fi

if [ -n "$load_exp" ]; then
    args="$args --load_exp $load_exp"
fi

if [ -n "$load_epoch" ]; then
    args="$args --load_epoch $load_epoch"
fi

if [ -n "$num_pertask" ]; then
    args="$args --num_pertask $num_pertask"
fi

if [ -n "$pretrain_data_ratio" ]; then
    args="$args --pretrain_data_ratio $pretrain_data_ratio"
fi

if [ -n "$fuse_method" ]; then
    args="$args --fuse_method $fuse_method"
fi

if [ -n "$encoder_lora_r" ]; then
    args="$args --encoder_lora_r $encoder_lora_r"
fi

if [ -n "$decoder_lora_r" ]; then
    args="$args --decoder_lora_r $decoder_lora_r"
fi

if [ -n "$lora_alpha" ]; then
    args="$args --lora_alpha $lora_alpha"
fi

if [ -n "$target_modules" ]; then
    args="$args --target_modules $target_modules"
fi

mkdir -p ./$meta_exp_dir/$exp_name

# 保存执行bash脚本的完整命令
echo "bash $0 $ORIGINAL_ARGS" > ./$meta_exp_dir/$exp_name/bash_command.txt

echo "$python_cmd $args" > ./$meta_exp_dir/$exp_name/terminal.txt

eval "$python_cmd $args" >> ./$meta_exp_dir/$exp_name/terminal.txt 2>&1
