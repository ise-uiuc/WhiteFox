
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # no-op

    def forward(self, x1):
        l1 = x1
        l2 = x1 + 3
        l3 = l2.clamp_min(0)
        l4 = l3.clamp_max(6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
