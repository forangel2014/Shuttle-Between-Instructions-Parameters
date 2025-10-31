# 获取transformers的安装路径
transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")

# 获取transformers的版本号
transformers_version=$(python -c "import transformers; print(transformers.__version__)")

echo $transformers_path
echo $transformers_version

# 使用版本号从当前目录的./版本号/下获取patch_files
python run_patch.py --package_path $transformers_path/models/llama --patch_filename modeling_llama.py --version $transformers_version
python run_patch.py --package_path $transformers_path/generation --patch_filename utils.py --version $transformers_version
python run_patch.py --package_path $transformers_path/models/qwen2 --patch_filename modeling_qwen2.py --version $transformers_version