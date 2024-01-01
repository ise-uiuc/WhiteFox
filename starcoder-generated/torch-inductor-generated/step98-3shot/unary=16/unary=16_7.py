
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.flatten(torch.nn.Linear(56, 56)(x1), (1, 2))
        v2 = torch.nn.ReLU()(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 8, 8)
