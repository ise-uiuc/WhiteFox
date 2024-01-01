
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=2)
        y = torch.cat([x, x, x], dim=0)
        return y
# Inputs to the model
x = torch.randn(1, 3)
