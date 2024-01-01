
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, False)

    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = torch.sigmoid(v1)
        v4 = v1 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
