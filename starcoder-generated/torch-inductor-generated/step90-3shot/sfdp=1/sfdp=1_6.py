
import torch
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_weight = torch.nn.parameter.Parameter(torch.zeros([size, size]))
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = qk.size(-1) ** -0.5
        dropout_qk = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1), p=dropout_p)
        ouput = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model(dim=10, num_heads=1, dropout_p=0.5)

# Inputs to the model
query = torch.randn(3, 50, 10)
key = torch.randn(3, 100, 10)
value = torch.randn(3, 100, 10)
output = m(query, key, value)

