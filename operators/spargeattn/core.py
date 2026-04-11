"""
SpargeAttention core APIs.

High-level Python APIs for block-sparse attention with quantized QK and V.
Supports SM80 (INT8 QK + FP16 SV) and SM90 (INT8 QK + FP8 SV with WGMMA + TMA).

Ported from SpargeAttn (https://github.com/thu-ml/SpargeAttn).
"""

import torch
from .utils import hyperparameter_check, get_block_map_meansim, get_block_map_meansim_fuse_quant, get_vanilla_qk_quant, block_map_lut_triton
from einops import rearrange

# Import CUDA extensions (built by setup.py)
import importlib
import importlib.util
import os
import sys

# Ensure torch is loaded first (needed for libtorch shared libraries)
import torch as _torch

def _load_extension(name):
    """Load a CUDA extension .so from the cuda/ directory."""
    # Use __file__ of this module to find the cuda/ directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_dir = os.path.join(this_dir, 'cuda')
    if not os.path.isdir(cuda_dir):
        return None
    # Search for the .so file
    for fname in os.listdir(cuda_dir):
        if fname.startswith(name) and fname.endswith('.so'):
            so_path = os.path.join(cuda_dir, fname)
            spec = importlib.util.spec_from_file_location(name, so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

try:
    qattn = _load_extension('_qattn')
    if qattn is None:
        raise ImportError("_qattn not found")
except Exception:
    qattn = None
    print("Warning: _qattn extension not found. Build with: cd cuda && python setup.py build_ext --inplace")

try:
    fused = _load_extension('_fused')
    if fused is None:
        raise ImportError("_fused not found")
except Exception:
    fused = None
    print("Warning: _fused extension not found. Build with: cd cuda && python setup.py build_ext --inplace")

# Check for SM90-specific functions
HAS_SM90 = False
if qattn is not None:
    HAS_SM90 = hasattr(qattn, 'qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90')


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


@torch.compiler.disable
def spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    """SpargeAttn based on SageAttention2 with CDF-based block selection."""
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # SM90: quantize V to FP8
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o


@torch.compiler.disable
def spas_sage2_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    """
    SpargeAttn based on SageAttention2 with top-K block selection.

    This is the **recommended plug-and-play API**.

    Args:
        q, k, v: Query, Key, Value tensors of shape (B, H, N, D) for HND layout
        topk: Fraction of K blocks to keep (0.5 = 50% sparsity). Higher = more accurate, lower = faster.
        is_causal: Whether to apply causal masking
        scale: Softmax scale (default: 1/sqrt(head_dim))
        smooth_k: Whether to subtract K mean for better quantization
        pvthreshd: PV threshold for skipping unimportant attention blocks
        tensor_layout: "HND" or "NHD"
        return_sparsity: If True, also return sparsity ratio
    """
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # SM90: quantize V to FP8
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o


@torch.compiler.disable
def block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    """
    Block-sparse attention with custom per-head mask.

    Args:
        q, k, v: Query, Key, Value tensors
        mask_id: Per-head attention mask of shape (B, H, ceil(N/BLKQ), ceil(N/BLKK)) with 0/1 values.
                 Block size is 128x64 for SM80, 64x128 for SM90.
    """
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]

    if arch == "sm90":
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # SM90: quantize V to FP8
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o


@torch.compiler.disable
def spas_sage_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    """SpargeAttn based on SageAttention (SM80 only, FP16 V)."""
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o


@torch.compiler.disable
def spas_sage_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    """SpargeAttn based on SageAttention (SM80 only, FP16 V) with top-K selection."""
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2) >= 128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
