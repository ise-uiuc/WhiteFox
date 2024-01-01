
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8, bias=True)

    def forward(self, x1):
        v1 = self.fc(x1)
        return torch.cat([v1], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
