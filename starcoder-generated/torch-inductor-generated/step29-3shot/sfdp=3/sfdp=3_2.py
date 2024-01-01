
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout=0.5, scale_factor=1/8):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout)
        out = v4.matmul(value)
        return out

# Initializing the model
m = Model()

# Inputs to the model
query = torch.rand(1, 64, 100)
key = torch.rand(1, 64, 200)
value = torch.rand(1, 64, 200)
