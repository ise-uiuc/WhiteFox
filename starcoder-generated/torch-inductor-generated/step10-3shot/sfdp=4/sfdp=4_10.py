
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
 
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
 
        self.scaling = self.head_dim ** -0.5
 
        self._qkv_projector = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self._output_projection = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, x, attn_mask=None):
        # x: tensor with shape {batch_size, seq_len, embed_dim}
        qkv = self._qkv_projector(x) # This is the projected queries, keys, and values. The exact value of "3*embed_dim" can be different for different models, so please refer to the usage document of your original PyTorch model for specific details.
        # qkv: tensor with shape {batch_size, seq_len, 3*embed_dim}
        # Separate the queries, keys, and values
        # q, k, v: each is a tensor with shape {batch_size, seq_len, embed_dim//num_heads}
        _batch_size, _seq_len, _embed_dim = qkv.shape
        qkv = qkv.reshape(_batch_size, _seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        # q: tensor with shape {batch_size, num_heads, seq_len, head_dim}
        # k: tensor with shape {batch_size, num_heads, head_dim, seq_len}
        # v: tensor with shape {batch_size, num_heads, head_dim, seq_len}
 
        # Compute the scaled dot product of the query and key, and scale it by the factor of the square root of the head dimension
        k_t = k.transpose(-2, -1)
        qk = torch.matmul(q, k_t) * self.scaling
        # qk: tensor with shape {batch_size, num_heads, seq_len, seq_len}
 
        # Add the attention mask to the result
        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
            ), f"`attn_mask` needs to have `dtype` torch.float32 (got {attn_mask.dtype})."
            qk = qk + attn_mask
 
        # Apply softmax to the result
        attn_weight = torch.softmax(qk, dim=-1)
        # attn_weight: tensor with shape {batch_size, num_heads, seq_len, seq_len}
 
        # Compute the weighted sum of the value
        attn_output = torch.matmul(attn_weight, v)
        # attn_output: tensor with shape {batch_size, num_heads, seq_len, head_dim}
 
        # Re-assemble the multi-head results
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(_batch_size, _seq_len, -1)
        # attn_output: tensor with shape {batch_size, seq_len, embed_dim}
 
        # Apply projection and return
        v1 = self._output_projection(attn_output) # This is an arbitrary transformation applied to attn_output.
        return v1

# Initializing the model
m = MultiHeadAttention(embed_dim=512, num_heads=8)

# Inputs to the model
x1 = torch.randn(1, 10, 512)
