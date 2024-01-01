
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, l0):
        l1 = self.linear(l0)
        l2 = l1 + 3
        l3 = torch.min(l2, torch.full_like(l2, 0, dtype=torch.float))
        l4 = torch.max(l3, torch.full_like(l3, 6, dtype=torch.float))
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
l0 = torch.randn(1, 3)
