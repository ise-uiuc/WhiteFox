
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        v1 = x1 * x2
        v2 = v1 + x3
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
x3 = torch.randn(1, 5, 64, 64)
