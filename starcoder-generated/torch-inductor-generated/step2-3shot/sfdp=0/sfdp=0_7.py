
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = v1 / 8.0
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.matmul(v3, x1) # Here, x1 is the value tensor
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 8)
x2 = torch.randn(1, 8, 16)
