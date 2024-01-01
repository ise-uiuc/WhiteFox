
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand((1024, 1024))
 
    def forward(self, x):
        v1 = torch.matmul(x, self.w)
        v2 = torch.clamp(v1, 0, 6) * v1 + 3
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1024)
