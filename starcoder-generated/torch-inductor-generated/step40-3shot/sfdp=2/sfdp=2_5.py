
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / inv_scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(dropout_qk, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(batch_size, num_queries, num_heads, head_size)
x2 = torch.randn(batch_size, num_keys, num_heads, head_size)
