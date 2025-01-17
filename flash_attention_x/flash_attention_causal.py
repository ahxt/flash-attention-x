import math
import torch
import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # switch device context
        orginal_device_index = torch.cuda.current_device()
        device_index = q.device.index
        torch.cuda.set_device(device_index)

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv
        assert Dk in {16, 32, 64, 128}
        
        BLOCK_M = 128
        BLOCK_N = 32
        num_stages = 2
        num_warps = 8

        B, H, M, D = q.shape
        N = k.shape[2]
        P_SEQ = N - M
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # consider using 3d grid to avoid div & rem
        # grid = (B, H, triton.cdiv(M, BLOCK_M), H, B)
        grid = (B, H, triton.cdiv(M, BLOCK_M))
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, M, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D, IS_CAUSAL=causal,
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = D
        ctx.P_SEQ = P_SEQ
        ctx.causal = causal

        # restore device context
        torch.cuda.set_device(orginal_device_index)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors

        # switching device context
        orginal_device_index = torch.cuda.current_device()
        device_index = q.device.index
        torch.cuda.set_device(device_index)

        B, H, M, D = q.shape
        N = k.shape[2]
        P_SEQ = N - M
        sm_scale = ctx.sm_scale
        causal = ctx.causal

        BLOCK_M = 64
        BLOCK_N = 64
        num_stages = 1
        
        delta = torch.empty_like(L)
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        _bwd_preprocess[grid](
            o, do,
            delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            M,
            BLOCK_M=BLOCK_M, D_HEAD=D,
        )

        dq = torch.zeros_like(q, dtype=torch.float32) # us float32 for atomic updates
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _bwd_kernel[grid](
            q, k, v, sm_scale, do, 
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q.shape[0], q.shape[1], q.shape[2], P_SEQ, 
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
            num_stages=num_stages,
        )
        dq = dq.to(q.dtype)
        torch.cuda.set_device(orginal_device_index)
        return dq, dk, dv, None, None, None




@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, 
    IS_CAUSAL: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    # start_m = tl.program_id(0)
    # off_h = tl.program_id(1)
    # off_z = tl.program_id(2)

    start_m = tl.program_id(2)
    off_h = tl.program_id(1)
    off_z = tl.program_id(0)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # log2e: tl.constexpr = 1.4426950408889634
    # qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * N_CTX # l's shape is (B, H, N_CTX)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    # initialize pointers to value-like data 
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    q = tl.load(q_ptrs, mask=mask_m[:, None])
    # Dot I trick: to place q in registers, it saves shared memory
    I = tl.where(offs_k[:, None] == offs_k,
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    q = tl.dot(q, I).to(input_dtype)

    if IS_CAUSAL:
        hi = P_SEQ + (start_m + 1) * BLOCK_M
    else:
        hi = N_CTX + P_SEQ

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        mask_n = offs_n < (N_CTX + P_SEQ)
        valid_mask = mask_m[:, None] & mask_n
        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
        # -- load k, v --
        k = tl.load(k_ptrs, mask=mask_n[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k, out_dtype=tl.float32)
        s *= sm_scale
        s = tl.where(valid_mask, s, float("-inf")) # be rigid & always apply mask
        if IS_CAUSAL:
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp(m_i - m_i_new )
        p = tl.math.exp(s - m_i_new[:, None])

        # -- scale and update acc: acc *= alpha[:, None]--
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    acc = acc / l_i[:, None]
    l = m_i + tl.math.log(l_i) # log(normalizer)
    tl.store(l_ptrs, l, mask=mask_m)
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None])



@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < M
    off_n = tl.arange(0, D_HEAD)
    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok
    do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    d_ptrs = Delta + off_m * stride_dm
    tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, N_CTX, P_SEQ, 
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    # offset pointers for batch/head
    D += (off_z * H + off_h) * N_CTX
    L += (off_z * H + off_h) * N_CTX

    if CAUSAL:
        lo = tl.math.max(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < (N_CTX + P_SEQ)

    # initialize pointers to value-like data 
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
    dq_ptrs = DQ + (offs_m_init[:, None] * stride_dqm + offs_k[None, :] * stride_dqk) # (BLOCK_M, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk) # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    v = tl.load(v_ptrs, mask=mask_n[:, None])
    k = tl.load(k_ptrs, mask=mask_n[:, None])

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # loop over a col
    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        mask_m = offs_m < N_CTX
        valid_mask = mask_m[:, None] & mask_n
        causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        q1 = tl.load(q_ptrs, mask=mask_m[:, None])
        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q1, tl.trans(k))

        # -- recompute p ---
        l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e) # (BLOCK_M, BLOCK_N)
        p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do) # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        delta = tl.load(D + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)
        ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds.to(input_dtype)

        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(tl.trans(ds), q1)

        # compute dq, update with atomic add
        dq = tl.dot(ds, k) * sm_scale
        tl.atomic_add(dq_ptrs, dq, mask=mask_m[:, None])

        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    dk *= sm_scale
    tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
    tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL,)


def flash_attn(q, k, v, causal=False, sm_scale=None):
    return FlashAttention.apply(q, k, v, causal, sm_scale)

def flash_attn_full(q, k, v, sm_scale=None):
    return FlashAttention.apply(q, k, v, False, sm_scale)

def flash_attn_causal(q, k, v, sm_scale=None):
    return FlashAttention.apply(q, k, v, True, sm_scale)





def pytorch_attention_full(q, k, v, sm_scale=None):
    input_dtype = q.dtype
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    S = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    P = torch.softmax(S, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(P.to(v.dtype), v)
    return attn_output.to(input_dtype)


def pytorch_attention_causal(q, k, v, causal=True, sm_scale=None):
    input_dtype = q.dtype
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)
    S = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        mask = torch.triu(torch.ones(S.shape[-2], S.shape[-1]), diagonal=1).to(S.device)
        S = S.masked_fill(mask == 1, float("-inf"))
    P = torch.softmax(S, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(P.to(v.dtype), v)
    return attn_output.to(input_dtype)




if __name__ == "__main__":
    import torch
    torch.manual_seed(0)

# query_states torch.Size([1, 32, 8, 128])
# key_states torch.Size([1, 32, 8, 128])
# value_states torch.Size([1, 32, 8, 128])
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, M, N, D = 1, 32, 8, 8, 128
    q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
    sm_scale = 1. / math.sqrt(D)

    o_ref = pytorch_attention_full(q.float(), k.float(), v.float(), sm_scale)
    o_ref.backward(do.float())
    dq_ref = q.grad
    dk_ref = k.grad
    dv_ref = v.grad
    q.grad = None
    k.grad = None
    v.grad = None

    o_pytorch = pytorch_attention_full(q, k, v, sm_scale)
    o_pytorch.backward(do)
    dq_pytorch = q.grad
    dk_pytorch = k.grad
    dv_pytorch = v.grad
    q.grad = None
    k.grad = None
    v.grad = None

    o_flash = flash_attn_full(q, k, v, sm_scale)
    o_flash.backward(do)
    dq_flash = q.grad
    dk_flash = k.grad
    dv_flash = v.grad

    def report(name, actual, expected):

        def max_diff(a, b):
            return (a - b).abs().max().item()

        def zero_percent(a, b):
            diff = (a - b).abs()
            num_non_zeros = diff.nonzero().shape[0]
            return (1.0 - num_non_zeros/ diff.numel()) * 100.0

        print(f"{name}: \tmax_diff: {max_diff(actual, expected):0.6f}\tzero_diff elements: {zero_percent(actual, expected):0.3f}%\tactual: {actual.shape}\texpected: {expected.shape}")

    report("o", o_flash, o_ref)
    report("dq", dq_flash, dq_ref)
    report("dk", dk_flash, dk_ref)
    report("dv", dv_flash, dv_ref)

    assert (o_ref - o_flash).abs().max().item() <= 2 * (o_ref - o_pytorch).abs().max().item()
    assert (dq_ref - dq_flash).abs().max().item() <= 2 * (dq_ref - dq_pytorch).abs().max().item()
    assert (dk_ref - dk_flash).abs().max().item() <= 2 * (dk_ref - dk_pytorch).abs().max().item()
    assert (dv_ref - dv_flash).abs().max().item() <= 2 * (dv_ref - dv_pytorch).abs().max().item()
