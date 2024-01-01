
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)

    def forward(self, x):
        l1 = self.linear(x)
        l2 = clamp(l1, min=0, max=6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 6)
