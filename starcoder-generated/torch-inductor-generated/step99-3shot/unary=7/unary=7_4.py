
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear =  torch.nn.Linear(224, 32)

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = clamp(l1, 0, 6)
        l3 = clamp(l1 + 3, 0, 6)
        l4 = l2 / 6
        return l4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(224)
