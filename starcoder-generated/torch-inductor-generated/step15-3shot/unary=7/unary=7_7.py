
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(192, 192)

    def forward(self, x):
        v1 = self.fc1(x)
        v2 = torch.clamp(v1, min=0, max=6)
        v3 = v2 + 3
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 192)
