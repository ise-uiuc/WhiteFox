
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU6())

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1)
