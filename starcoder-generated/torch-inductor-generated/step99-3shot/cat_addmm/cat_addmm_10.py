
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.cat([x, x], dim=1)
        y = x + x
        return y
# Inputs to the model
x = torch.randn(2, 1)
