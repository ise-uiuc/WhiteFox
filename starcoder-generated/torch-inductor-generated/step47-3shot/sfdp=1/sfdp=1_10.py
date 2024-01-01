
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, v3):
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(v3)
        v6 = torch.softmax(v5, dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.5)
        v8 = torch.matmul(v7, v3)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 10, 20)
v2 = torch.randn(1, 20, 30)
v3 = torch.arange(1, 20 + 1, dtype=torch.float32)
