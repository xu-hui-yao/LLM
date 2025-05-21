import torch
import huggingface_hub
from transformers import AutoModel

with open('huggingface_secrets.txt', 'r') as f:
    hf_secrets = f.read()
    huggingface_hub.login(token=str(hf_secrets))


class TokenEmbedding(torch.nn.Module):
    """Expands pretrained token embeddings while preserving original weights.

    Designed for language model expansion scenarios where new tokens need to be
    added while maintaining pretrained embeddings for existing vocabulary.

    Args:
        vocab_size (int): Total vocabulary size including new tokens
        padding_idx (int): Index used for padding tokens
        model_name (str): Pretrained model identifier
        freeze_embeddings (bool): Whether to freeze original embeddings
    """

    def __init__(self, vocab_size: int, padding_idx: int,
                 model_name: str = 'meta-llama/Llama-3.1-8B',
                 freeze_embeddings: bool = True):
        super().__init__()

        # Load pretrained model on CPU to save GPU memory
        pretrained_model = AutoModel.from_pretrained(model_name)
        pretrained_embedding = pretrained_model.get_input_embeddings()

        # Get embedding dimensions from pretrained model
        pretrained_vocab_size, embedding_dim = pretrained_embedding.weight.shape
        self.pretrained_vocab_size = pretrained_vocab_size

        # Initialize new embedding layer
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # Parameter initialization strategy
        with torch.no_grad():
            # 1. Copy pretrained embeddings
            self.embedding.weight[:pretrained_vocab_size] = pretrained_embedding.weight.to(self.embedding.weight.device)

            # 2. Initialize new tokens using pretrained distribution
            pretrained_mean = pretrained_embedding.weight.mean(dim=0)
            pretrained_std = pretrained_embedding.weight.std(dim=0)
            new_vectors = torch.randn(
                vocab_size - pretrained_vocab_size,
                embedding_dim,
                device=self.embedding.weight.device
            ) * pretrained_std + pretrained_mean

            self.embedding.weight[pretrained_vocab_size:] = new_vectors

            # 3. Ensure padding token is zero-initialized
            self.embedding.weight.data[padding_idx] = torch.zeros_like(self.embedding.weight[padding_idx])

        # Freeze original embeddings if requested
        if freeze_embeddings:
            self.embedding.weight.requires_grad_(False)
            # Unfreeze new tokens if we're freezing original embeddings
            if vocab_size > pretrained_vocab_size:
                self.embedding.weight.data[pretrained_vocab_size:].requires_grad_(True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed input tokens with combined vocabulary support.

        Args:
            tokens (Tensor): Input token indices [batch_size, seq_len]

        Returns:
            Tensor: Embedded sequence [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(tokens)


class RotaryPositionalEmbedding(torch.nn.Module):
    """Implements Rotary Position Embedding (RoPE) for transformer models.

    This encoding technique injects positional information using rotation matrices
    in a way that preserves relative position relationships through dot product attention.

    Args:
        dim (int): Dimension of each attention head's features
        seq_len (int): Length of sequence.
        theta (float): Hyper parameter
    """

    def __init__(self, dim: int, seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.theta = theta
        # Precompute inverse frequencies for positional encoding, shape: [dim // 2]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        t = torch.arange(self.seq_len)
        freqs = torch.outer(t, inv_freq).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies rotary positional encoding to input tensor.

        Args:
            x (Tensor): Input features [batch, heads, seq_len, dim]

        Returns:
            Tensor: Position-aware features with same shape as input
        """
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)  # [batch, heads, seq_len, dim // 2, 2]
        x_ = torch.view_as_complex(x_)
        x_out = torch.view_as_real(x_ * self.freqs_cis).flatten(3)
        return x_out.type_as(x)


class MaskedMultiHeadAttention(torch.nn.Module):
    """Implements multi-head self attention with RoPE and causal masking.

    Args:
        d_model (int): Total dimension of the model
        num_heads (int): Number of attention heads
        seq_len (int): Length of sequence
    """

    def __init__(self, d_model: int, num_heads: int, seq_len: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.seq_len = seq_len

        # Projection matrices
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        # Rotary positional embedding per head
        self.rope = RotaryPositionalEmbedding(self.head_dim, self.seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with causal attention and positional encoding.

        Args:
            x (Tensor): Input sequence [batch, seq_len, d_model]
            mask (Tensor, optional): Custom attention mask

        Returns:
            Tensor: Output sequence [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections (split into heads later)
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head format
        # New shape: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position encoding to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # Compute attention scores
        # [batch, heads, seq_len, seq_len]
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))

        # Apply causal mask if not provided
        if mask is None:
            # Create lower triangular mask (prevents looking ahead)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Apply mask (broadcast across batch and heads)
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

        # Normalize with softmax
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # [batch, heads, seq_len, head_dim]
        output = attn_weights @ v

        # Merge heads and project back to model dimension
        # [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

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

    def __init__(self, d_model, num_heads, hidden_dim, seq_len):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(d_model, num_heads, seq_len)
        self.ffn = FeedForward(d_model, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.norm1(self.attn(x))
        # FFN with residual
        x = x + self.norm2(self.ffn(x))
        return x


class Transformer(torch.nn.Module):
    """Full Decoder-only Transformer"""

    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, hidden_dim=2048, seq_len=1024):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim, seq_len)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
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
