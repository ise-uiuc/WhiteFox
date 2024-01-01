
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, l1):
        return clamp(min=0, max=6, l1 + 3) * l1 / 6

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 100)
