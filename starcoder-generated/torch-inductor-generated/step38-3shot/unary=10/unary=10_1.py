
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 16)

    def forward(self, x1):
        l1 = self.linear1(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
nn = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
