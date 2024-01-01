
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 100)
    def forward(self, x):
        x = self.layers(x)
        x = x.mul(x)
        x = torch.cat([x, x, x, x, x, x], dim=0)
        x = x + x
        x = x.add(x)
        return x
# Inputs to the model
x = torch.randn(1, 100)
