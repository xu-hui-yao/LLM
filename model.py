import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        # x: [batch, heads, seq_len, dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

        # Expand to match dimensions
        emb = emb[None, None, :, :]  # [1, 1, seq_len, dim]
        return x * emb.cos() + self._rotate_half(x) * emb.sin()


class MaskedMultiHeadAttention(torch.nn.Module):
    """Multi-head self attention with RoPE and causal masking"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(self.head_dim)

        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

        # Softmax and attention output
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = attn_weights @ v  # [batch, heads, seq_len, head_dim]

        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(output)


class FeedForward(torch.nn.Module):
    """Position-wise Feed Forward Network"""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    """Single Transformer Decoder Layer"""

    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(torch.nn.Module):
    """Full Decoder-only Transformer"""

    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, hidden_dim=2048):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, d_model]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits
