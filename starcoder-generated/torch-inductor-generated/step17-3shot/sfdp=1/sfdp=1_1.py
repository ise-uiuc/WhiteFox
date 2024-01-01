
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        v3 = torch.matmul(query, key.transpose(-2, -1))
        v5 = v3.div(inv_scale_factor)
        v6 = torch.nn.functional.dropout(v5, p=dropout_p)
        v7 = torch.matmul(v6, value)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, seq_len, num_heads, head_dim)
key = torch.randn(batch_size, seq_len, num_heads, head_dim)
value = torch.randn(batch_size, seq_len, num_heads, head_dim)
inv_scale_factor = torch.randn(batch_size, num_heads, seq_len, seq_len)
dropout_p = 0.5
