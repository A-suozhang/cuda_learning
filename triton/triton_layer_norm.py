import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    # Mean,  # pointer to the mean
    # Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float16)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float16)
        x = tl.where(cols < N, x - mean, 0.).to(tl.float16)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # # Write mean / rstd
    # tl.store(Mean + row, mean)
    # tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float16)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

class LayerNorm(torch.autograd.Function):

    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        # x_arg = x.reshape(-1, x.shape[-1])
        x_arg = x
        M, N = x_arg.shape
        # mean = torch.empty((M, ), dtype=torch.float16, device=x.device)
        # rstd = torch.empty((M, ), dtype=torch.float16, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        return y


"""
Example Triton kernel that implements fused Layenorm + Quantization.
Also performs layout conversion from row-major to COL32.
The kernel code is adapted from the Triton Lang tutorial.
See https://triton-lang.org/master/getting-started/tutorials/05-layer-norm.html
"""
import triton
import triton.language as tl

@triton.jit
def layernorm_fused_quant(
    Input,
    Output,
    Weight,
    Bias,
    QuantScale,
    stride, # Stride between rows
    M, # Number of rows
    N, # Number of columns
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # stride_out = 32 # Because COL32
    stride_out = stride # INFO: when using col32 format, the pytorch all_close will incur error

    # Position of elements processed by this program
    row = tl.program_id(0)
    Output += row * stride_out
    Input += row * stride

    # Layenorm: Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(Input + cols, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float16)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Layernorm: Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(Input + cols, mask=cols < N, other=0.0, eviction_policy="evict_last").to(tl.float16)
        a = tl.where(cols < N, a - mean, 0.0).to(tl.float16)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Layernorm: Multiply by weight, and add bias
    _global_max = tl.zeros([], dtype=tl.float16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(Input + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float16)
        a_hat = (a - mean) * rstd
        y = a_hat * weight + bias
        _y_max = tl.max(tl.abs(y), axis=0).to(tl.float16)
        if _y_max > _global_max:
            _global_max = _y_max
            # print(f'updated _global_max {_global_max} at step {off}') # DEBUG_ONLY
        # else:
            # print(f'Cur _y_max {_y_max} smaller than {_global_max} at step {off}')
        tl.store(Output + cols, y, mask=mask)

    # Apply Quantize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # static pre-defined scale
        # quant_scale = tl.load(QuantScale + cols, mask=mask)

        # dynamic calculated scale
        quant_scale = tl.full([BLOCK_SIZE], _global_max, dtype=tl.float16)

        pos_clamps = tl.zeros([BLOCK_SIZE], dtype=tl.float16) + 127
        neg_clamps = tl.zeros([BLOCK_SIZE], dtype=tl.float16) - 127

        y = tl.load(Output + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float16)
        y = (y * quant_scale).to(tl.float32)
        y = tl.extra.cuda.libdevice.rint(y)  # only support fp32
        y = tl.where(y > 127, pos_clamps, y)
        out = tl.where(y < -127, neg_clamps, y)

        # Pointer arithmetic for Row-major --> COL32
        # cols_out = cols // stride_out * (stride_out * M) + (cols % stride_out)

        # Store output
        # tl.store(Output + cols_out, out, mask=mask)

        tl.store(Output + cols, out, mask=mask)

class LayerNormWithQuant(torch.autograd.Function):

    def forward(ctx, x, normalized_shape, weight, bias, quant_scale_value, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float16, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float16, device=x.device)
        # quant_scale = torch.full_like(x, quant_scale_value, dtype=torch.float16, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        layernorm_fused_quant[(M, )](  #
            x_arg, y, weight, bias, quant_scale_value,  #
            x_arg.stride(0), M, N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        return y
