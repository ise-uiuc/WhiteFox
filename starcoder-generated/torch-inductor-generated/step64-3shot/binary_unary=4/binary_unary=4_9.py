
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is not None:
            v1 = v1 + other
        v1 = F.relu(v1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
