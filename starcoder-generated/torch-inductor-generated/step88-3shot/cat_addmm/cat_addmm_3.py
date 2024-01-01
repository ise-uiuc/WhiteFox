
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=1).flatten(1)
        y = torch.stack((y, y, y, y), dim=1).flatten(1)
        return x + y
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
