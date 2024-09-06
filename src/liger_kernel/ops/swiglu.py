import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):

    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b=None):
        if b is None:
            ctx.b_is_none = True
            return LigerSiLUMulFunctionMergedInput.forward(ctx, a)
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        if hasattr(ctx, "b_is_none") and ctx.b_is_none:
            return LigerSiLUMulFunctionMergedInput.backward(ctx, dc), None
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b

@triton.jit
def _swiglu_forward_kernel_merged_input(
    input_ptr, output_ptr, input_b_offset: tl.constexpr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # locate start index
    pid = tl.program_id(0)
    input_a_ptr = input_ptr + (pid * n_cols * 2)
    input_b_ptr = input_ptr + (pid * n_cols * 2) + input_b_offset
    output_ptr += pid * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    input_a_row = tl.load(input_a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    input_b_row = tl.load(input_b_ptr + col_offsets, mask=mask, other=0)
    output_row = silu(input_a_row) * input_b_row
    tl.store(output_ptr + col_offsets, output_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel_merged_input(
    d_output_ptr, input_ptr, input_b_offset: tl.constexpr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # locate start index
    pid = tl.program_id(0)
    d_output_ptr += pid * n_cols
    input_a_ptr = input_ptr + (pid * n_cols * 2)
    input_b_ptr = input_ptr + (pid * n_cols * 2) + input_b_offset

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    d_output_row = tl.load(d_output_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    input_a_row = tl.load(input_a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    input_b_row = tl.load(input_b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(input_a_row)
    silu_a = input_a_row * sig_a
    d_input_b_row = d_output_row * silu_a
    d_input_a_row = d_output_row * (silu_a * (1 - sig_a) + sig_a) * input_b_row

    tl.store(input_a_ptr + col_offsets, d_input_a_row, mask=mask)
    tl.store(input_b_ptr + col_offsets, d_input_b_row, mask=mask)

class LigerSiLUMulFunctionMergedInput(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input):
        ori_shape = input.shape
        ctx.ori_shape = ori_shape
        assert len(ori_shape) in [2, 3]
        input = input.view(-1, ori_shape[-1])
        n_rows = input.shape[0]
        n_cols = input.shape[-1]//2
        output = torch.empty((n_rows, n_cols), dtype=input.dtype, device=input.device, requires_grad=True)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        _swiglu_forward_kernel_merged_input[(n_rows,)](
            input,
            output,
            input_b_offset=n_cols,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(input)
        return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, d_output):

        n_rows = d_output.shape[0]
        n_cols = d_output.shape[-1]
        d_output = d_output.view(-1, n_cols)
        input = ctx.saved_tensors[0]

        BLOCK_SIZE, num_warps = calculate_settings(n_cols*2)

        _swiglu_backward_kernel_merged_input[(n_rows,)](
            d_output,
            input,
            input_b_offset=n_cols,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return input if len(input.shape) == 3 else input.view(*ctx.ori_shape)
