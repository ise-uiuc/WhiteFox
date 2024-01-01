
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.Sequential(torch.nn.Linear(10, 5),
                                           torch.nn.Linear(5, 1))

    def forward(self, x1):
        l1 = self.linears(x1)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        return l3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10)
