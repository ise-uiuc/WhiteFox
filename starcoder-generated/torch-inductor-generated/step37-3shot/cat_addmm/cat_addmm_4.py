
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 10)
    def forward(self, x):
        x = torch.stack((x, x, x), dim=1)
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        y = torch.stack((x, x, x, x, x), dim=1)
        z = y
        return z
# Inputs to the model
x = torch.randn(2, 5)
