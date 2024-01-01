
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        t1 = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = t1.div(10.0)
        v2 = v1.softmax(-1)
        v3 = torch.nn.functional.dropout(v2, p=0.0)
        v4 = torch.matmul(v3, x3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10, 64)
x2 = torch.randn(64, 32)
x3 = torch.randn(32, 16)
