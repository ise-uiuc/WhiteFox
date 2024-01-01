
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        l1 = torch.randn(12, 23)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(23)
