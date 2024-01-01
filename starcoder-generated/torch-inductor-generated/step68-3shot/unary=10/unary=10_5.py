
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, l1):
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 1, 64, 64)
