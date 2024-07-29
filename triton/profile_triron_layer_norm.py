import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example usage
seed_everything(42)#

# import os
# os.environ["TRITON_INTERPRET"]="1"

from triton_layer_norm import LayerNorm, LayerNormWithQuant

M = 2096 # num_of_token
N = 1024  # num_of_channel
dtype = torch.float16
device = 'cuda'
eps = 1.e-5
x_shape = (M, N)
w_shape = (x_shape[-1],)
weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

layer_norm_triton = LayerNorm.apply
layer_norm_q_triton = LayerNormWithQuant.apply

# quant_utils
quant_scale_value = torch.full_like(x, 256 / (x.abs().max()), dtype=dtype, device=device)
# Noted that we apply this scale to 'y' after layer norm, but not x, so clipping will occur. 


# the nvidia rint implementation: round x to the nearest integer value in floating-point format, with halfway cases rounded to the nearest even integer value.
def quant_op_pytorch(x, quant_scale_value, dynamic=False):
    if dynamic:
        quant_scale_value = torch.max(x.abs(), dim=1)[0].unsqueeze(1)  # [M, 1]

    x = x * quant_scale_value
    rounded = torch.round(x)

    # Identify halfway cases (rare)
    # halfway_cases = (x % 1 == 0.5) | (x % 1 == -0.5)
    # # For halfway cases, adjust to the nearest even integer
    # even_adjustment = (rounded % 2 != 0).float()
    # even_adjustment[even_adjustment == 1] = -1
    # rounded[halfway_cases] += even_adjustment[halfway_cases]

    rounded_clip = torch.clamp(rounded, -127, 127)
    return rounded_clip

def run_once():
    y_triton = layer_norm_triton(x, w_shape, weight, bias, eps)
    y_triton_q = layer_norm_q_triton(x, w_shape, weight, bias, quant_scale_value, eps)

    y_pytorch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    y_pytorch_q = quant_op_pytorch(y_pytorch, quant_scale_value, dynamic=True)

    # assert torch.allclose(y_triton_q, y_pytorch_q, atol=1.e-2, rtol=0)
    try:
        assert torch.allclose(y_triton_q, y_pytorch_q, atol=1., rtol=0)
    except:
        import ipdb; ipdb.set_trace()
    # assert torch.allclose(y_triton_q, y_pytorch_q, atol=1.e-2, rtol=0)

run_once()

# ---------------------
# Profile Settings
PROFILE_ON = True

if PROFILE_ON:
    import nvtx
    import os

    for i in range(5):
        if i <= 2:
            run_once()
        else:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():

                    with nvtx.annotate('layer_norm_triron_{}'.format(i-2)):
                        y_triton = layer_norm_triton(x, w_shape, weight, bias, eps)
                    with nvtx.annotate('layer_norm_triron_q_{}'.format(i-2)):
                        y_triton_q = layer_norm_q_triton(x, w_shape, weight, bias, quant_scale_value, eps)
                    with nvtx.annotate('layer_norm_pytorch_{}'.format(i-2)):
                        y_pytorch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
                    with nvtx.annotate('quant_op_pytorch_{}'.format(i-2)):
                        y_pytorch_q = quant_op_pytorch(y_pytorch, quant_scale_value)
                    assert torch.allclose(y_triton_q, y_pytorch_q, atol=1.e-2, rtol=0)



