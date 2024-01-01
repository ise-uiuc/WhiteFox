
class Model(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, scaling_factor: float = None):
        super().__init__()
        if scaling_factor is None:
            assert embed_dim % num_heads == 0, "`embed_dim` must be divisible by `num_heads`"
            self._scaling_factor = float(embed_dim // num_heads)
        else:
            self._scaling_factor = float(scaling_factor)
 
    def forward(self, query, key, value, dropout_p):
        qk = query.matmul(key.transpose(-1, -2))
        scaled_qk = qk.div(self._scaling_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model(8, 512)

# Inputs to the model
query = torch.randn(1, 8, 26, 512)
key = torch.randn(1, 8, 26, 512)
value = torch.randn(1, 8, 26, 512)
dropout_p = 0.1
