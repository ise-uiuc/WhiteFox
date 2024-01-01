
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(16)
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 1, 32)
x2 = torch.randn(4, 32, 10)
x3 = torch.randn(4, 10, 64)
