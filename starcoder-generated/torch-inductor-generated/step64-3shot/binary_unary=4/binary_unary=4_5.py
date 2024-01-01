
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = torch.nn.Parameter(other)

    def forward(self, x1):
        v1 = x1 @ torch.randn(5, 10)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.zeros((5,)))

# Inputs to the model
x1 = torch.randn(1, 10)
