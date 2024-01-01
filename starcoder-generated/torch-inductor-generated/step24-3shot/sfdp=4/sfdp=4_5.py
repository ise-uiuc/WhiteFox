
class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
    
    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.depth)
        # Separate (linear transformation)
        return x.transpose(1, 2).contiguous()

    def forward(self, q, k, v, mask):
        # Linearly project the query, key and value tensors into multi-headed query, key and value tensors
        query = self._split_heads(q)
        key = self._split_heads(k)
        value = self._split_heads(v)

        # Scale q and k by sqrt of d_model
        query = query * (int(self.d_model) ** -0.5)

        # Scale q and k by sqrt of d_model
        key = key * (int(self.d_model) ** -0.5)

        # Compute the attention weights
        # attn_logits = (B, h, T, T) = (B, h, T, T) + (B, h, T, T)
        attn_logits = torch.matmul(query, key.transpose(-2, -1))

        # Optionally apply the attention mask
        if mask is not None:
            attn_logits = attn_logits + mask

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Compute the attention vectors
        # attn = (B, h, T, D) = (B, h, T, T) @ (B, h, T, D)
        attn = torch.matmul(attn_weights, value)
        attn = attn.transpose(1, 2).contiguous()
        # Combine attn tensors
        # attn = (B, T, 8) = (B, h, T, D) @ (B, h, D, T)
        attn = attn.view(B, -1, self.num_heads * self.depth)
        output = attn

        return output

# Initializing the model
embed_dim = 8
num_heads = 4    # Number of heads
d_model = embed_dim * num_heads
num_steps = 64  # Maximum input sequence length
m = MultiheadAttention(d_model, num_heads)

# Inputs to the model
q = torch.randn(4, 5, embed_dim)    # Queries. (B, T, dim)
k = torch.randn(4, 4, embed_dim)    # Keys. (B, T, dim)
v = torch.randn(4, 4, embed_dim)    # Values. (B, T, dim)

# Padding mask. This ensures that attention is not applied to the padding area of the attention layer
mask = torch.triu(torch.ones(q.size(1), k.size(1)) * float('-inf'), diagonal=1) > 0

# Outputs of the model with torch.Tensor input
m(q, k, v, mask)
# Outputs of the model with torch.nn.init.Constant generated input
m(torch.nn.init.Constant(torch.randn(4, 5, embed_dim)), 
   torch.nn.init.Constant(torch.randn(4, 4, embed_dim)),
   torch.nn.init.Constant(torch.randn(4, 4, embed_dim)),
   torch.nn.init.Constant(torch.randn((q.size(1), k.size(1)))))

