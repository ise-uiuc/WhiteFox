
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2)
        v2 = v1 * 0.7071067811865476
        v3 = v1 * 0.5
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2.matmul(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
