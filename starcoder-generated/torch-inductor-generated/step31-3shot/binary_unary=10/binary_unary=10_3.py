
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(56, 32, bias=False)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones_like(v1) / 56
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 56)
