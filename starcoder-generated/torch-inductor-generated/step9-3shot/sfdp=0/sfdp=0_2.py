
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
        # `query` has shape (b, n_heads, seq_len, head_dim)
        # `key` has shape (b, n_heads, seq_len, head_dim)
        # `value` has shape (b, n_heads, seq_len, head_dim)
        # `inv_scale` has shape (b, n_heads, 1, 1)
query = torch.randn(4, 2, 8, 16)
key = torch.randn(4, 2, 8, 16)
value = torch.randn(4, 2, 8, 16)
inv_scale = torch.tensor([[[[16.0]], [[16.0]]]])
__outputs__ = m(query, key, value, inv_scale)