import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import torch_utils.custom_ops as custom_ops
custom_ops.verbosity = 'full'

print("--- Testing bias_act_plugin ---")
try:
    sources = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch/torch_utils/ops/bias_act.cpp'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch/torch_utils/ops/bias_act.cu'),
    ]
    custom_ops.get_plugin('bias_act_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
    print("bias_act_plugin: SUCCESS")
except Exception as e:
    print(f"bias_act_plugin: FAILED\n{e}")

print("\n--- Testing upfirdn2d_plugin ---")
try:
    sources = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch/torch_utils/ops/upfirdn2d.cpp'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch/torch_utils/ops/upfirdn2d.cu'),
    ]
    custom_ops.get_plugin('upfirdn2d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
    print("upfirdn2d_plugin: SUCCESS")
except Exception as e:
    print(f"upfirdn2d_plugin: FAILED\n{e}")
