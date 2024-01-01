
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x2, x3):
        v1 = torch.matmul(x2, x3.transpose(-2, -1))
        v2 = v1.mul(0.125)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.14285714924716949)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(4, 2, 16)
x3 = torch.randn(4, 16, 8)
