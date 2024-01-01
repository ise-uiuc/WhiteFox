
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.unsqueeze(x, 0)
        x = torch.flatten(x)
        return x
# Inputs to the model
x = torch.randn(4)

# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.unsqueeze(x, 1)
        return x
# Inputs to the model
x = torch.randn(4, 1)
