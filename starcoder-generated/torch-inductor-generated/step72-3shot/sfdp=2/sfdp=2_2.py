
class Model(torch.nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 value_dim,
                 num_heads,
                 dropout_p=0.0):
        super().__init__()
        self.head_dim = query_dim // num_heads
        assert key_dim == value_dim
        assert self.head_dim == value_dim // num_heads
        self.num_heads = num_heads

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        # Unpack the shape of the query tensor
        __query_height__, __query_width__, __query_channels__ = query.shape[-3:]

        # Pre-process the query tensor and the key tensor
        query = query.reshape(-1, __query_height__, __query_width__, self.num_heads, self.head_dim)
        key = key.reshape(-1, __query_height__, __query_width__, self.num_heads, self.head_dim)

        # Compute the dot product of the query and the key
        qk = torch.matmul(query, key.transpose(-2, -1))

        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(inv_scale_factor)

        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)

        # Pre-process the value tensor
        value = value.reshape(-1, __query_height__, __query_width__, self.num_heads, self.head_dim)

        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(value)
        output = output.reshape(-1, __query_height__, __query_width__, self.num_heads * self.head_dim)
        return output

# Initializing the model
m = Model(query_dim=64,
          key_dim=64,
          value_dim=64,
          num_heads=2)

# Inputs to the model
query = torch.randn(1, 4, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
inv_scale_factor = torch.rand(1)
dropout_p = 0.5
