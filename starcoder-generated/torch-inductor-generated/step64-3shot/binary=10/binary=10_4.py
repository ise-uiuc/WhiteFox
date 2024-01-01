
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
other = torch.randn(1, 2)
