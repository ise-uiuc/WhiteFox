
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 100 + 100)
        self.layers1 = nn.Linear(100, 100 + 100)
        self.layers2 = nn.Linear(100, 100 + 100)
        self.layers3 = nn.Linear(100, 100 + 100)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x), dim=1)
        x = x.flatten(start_dim=1)
        x = (self.layers1(x), self.layers2(x), self.layers3(x))
        return x
# Inputs to the model
x = torch.randn(5, 100)
