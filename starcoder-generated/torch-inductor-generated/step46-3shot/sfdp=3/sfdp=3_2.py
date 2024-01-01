
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(dropout_q, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 3, 32, 5)
key = torch.randn(8, 6, 32, 10)
value = torch.randn(8, 6, 32, 10)
scale_factor = torch.tensor([0.0], dtype=torch.float)
dropout_p = torch.tensor([0.1], dtype=torch.float)
