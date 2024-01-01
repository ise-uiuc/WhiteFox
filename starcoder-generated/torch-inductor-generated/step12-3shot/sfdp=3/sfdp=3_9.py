
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 4, 5)
key = torch.randn(2, 4, 5, 6)
value = torch.randn(2, 4, 5, 6)
scale_factor = 1e5
dropout_p = 0.1
v1 = m.forward(query, key, value, scale_factor, dropout_p)
