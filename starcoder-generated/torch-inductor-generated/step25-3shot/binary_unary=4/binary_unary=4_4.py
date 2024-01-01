
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(25, 1, bias=False)

    def forward(self, x1, other=None):
        if other is None:
            other = torch.ones_like(x1[:, -1, :, :])
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 25, 1, 1)
