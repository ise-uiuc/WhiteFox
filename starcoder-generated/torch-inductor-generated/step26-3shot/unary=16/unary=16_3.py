 for pattern 0
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(240, 720)

    def forward(self, x1):
        v1 = self.linear(x1.view(-1, 60 * 240))
        v2 = torch.relu(v1)
        return v2

# Initializing the model for pattern 0
m0 = Model()

# Inputs to the model for pattern 0
x1 = torch.randn(1, 3, 64, 64)
