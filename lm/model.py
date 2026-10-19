import math
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        w = torch.empty(out_features, in_features)
        nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        return einsum(x, self.weight, "... i, o i -> ... o")


class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(w, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(w)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).to(x.dtype) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        a = self.w1(x)
        return self.w2(a * torch.sigmoid(a) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta, d_k, max_seq_len):
        super().__init__()
        theta_i = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k) # (d_k/2,)
        m = torch.arange(max_seq_len, dtype=torch.float32)                       # (max_seq_len,)
        ang = torch.outer(m, theta_i)                                            # (max_seq_len, d_k/2)
        self.register_buffer("cos_cached", ang.cos(), persistent=False)
        self.register_buffer("sin_cached", ang.sin(), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos_cached[token_positions] # (..., seq, d_k/2)
        sin = self.sin_cached[token_positions]
        x1 = x[..., 0::2].float()
        x2 = x[..., 1::2].float()
        y = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return y.to(x.dtype)


def softmax(x, dim):
    m = x.max(dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q.float() @ K.float().transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return (softmax(scores, dim=-1) @ V.float()).to(Q.dtype)


class MultiheadSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, num_kv_heads=None, rope_theta=None, max_seq_len=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_head)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.d_head)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.d_head)
        self.output_proj = Linear(num_heads * self.d_head, d_model)
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_head, max_seq_len)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        *lead, seq, _ = x.shape
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device).expand(*lead, seq)
            pos = token_positions.unsqueeze(-2) # (..., 1, seq)
            q = self.rope(q, pos.expand(*lead, self.num_heads, seq))
            k = self.rope(k, pos.expand(*lead, self.num_kv_heads, seq))

        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=-3) #rep=2: k1 k2 k3 k4 -> k1 k1 k2 k2 k3 k3 k4 k4 
            v = v.repeat_interleave(rep, dim=-3)

        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, "... h s d -> ... s (h d)")
        return self.output_proj(o)


class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, num_kv_heads=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, num_kv_heads=num_kv_heads,
                                           rope_theta=theta, max_seq_len=max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions=None):
        h = x + self.attn(self.ln1(x), token_positions)
        return h + self.ffn(self.ln2(h))


class TransformerLM(nn.Module):
    
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 rope_theta=10000.0, num_kv_heads=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x).float()


def count_params(vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 num_kv_heads=None, tied=False):
    hd = d_model // num_heads
    nkv = num_kv_heads or num_heads
    attn = d_model * d_model + 2 * (nkv * hd) * d_model + d_model * d_model
    block = attn + 3 * d_model * d_ff + 2 * d_model
    total = num_layers * block + d_model + vocab_size * d_model
    if not tied:
        total += vocab_size * d_model
    return total


def count_flops(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, seq_len,
                num_kv_heads=None):
    hd = d_model // num_heads
    nkv = num_kv_heads or num_heads
    attn = d_model * d_model + 2 * (nkv * hd) * d_model + d_model * d_model
    matmul_params = num_layers * (attn + 3 * d_model * d_ff) + vocab_size * d_model
    fwd = 2 * matmul_params + 4 * num_layers * seq_len * d_model
    return fwd, 3 * fwd


def cross_entropy(inputs, targets):
    inputs = inputs.float()
    z = inputs - inputs.max(-1, keepdim=True).values
    lse = z.exp().sum(-1).log()
    tgt = z.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - tgt).mean()


def clip_grad_norm_(parameters, max_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef = max_norm / (total + eps)
    if coef < 1:
        for g in grads:
            g.mul_(coef)
    return total


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]
                p.mul_(1 - lr * wd)                          # decoupled weight decay
                m.lerp_(g, 1 - b1)                           # m = b1*m + (1-b1)*g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)      # v = b2*v + (1-b2)*g^2
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


def lr_cosine(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        phase = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(phase))
    return min_learning_rate


def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[i: i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([dataset[i + 1: i + context_length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


@torch.no_grad()
def generate(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
             weights, prompt_ids, max_new_tokens, temperature, top_p, eos_id, seed, device="cpu"):
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()                                       # device="cuda" on Colab
    gen = torch.Generator(device=device).manual_seed(seed)

    ids = list(prompt_ids)
    new = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1]                                  # (vocab_size,)
        probs = softmax(logits / temperature, dim=-1)
        # nucleus: smallest set of top tokens whose cumulative probability reaches top_p
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        k = int((cum < top_p).sum().item()) + 1
        keep = si[:k]
        filtered = torch.zeros_like(probs)
        filtered[keep] = probs[keep]
        filtered = filtered / filtered.sum()
        nxt = int(torch.multinomial(filtered, num_samples=1, generator=gen).item())
        ids.append(nxt)
        new.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return new
