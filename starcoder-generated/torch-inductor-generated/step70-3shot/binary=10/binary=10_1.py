
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.nn.Linear(3, 64, bias=None)(x1)
        return v1 + x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 3, 64, 64)
x2 = torch.randn(64, 4, 64, 64)
