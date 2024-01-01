
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.layers2 = nn.Linear(2, 4)
    def forward(self, x, y):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        y = self.layers2(y)
        y = torch.stack((y, y), dim=1)
        z = x + y
        z = z.view(41, 1, 1)
        return z
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(100, 2)
