
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 32)
        self.other = other

    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
other = torch.randn(32)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 64)
