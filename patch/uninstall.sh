py_package_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo $py_package_path
python remove_patch.py --package_path $py_package_path/transformers/models/llama --patch_files modeling_llama.py
python remove_patch.py --package_path $py_package_path/transformers/generation --patch_files utils.py