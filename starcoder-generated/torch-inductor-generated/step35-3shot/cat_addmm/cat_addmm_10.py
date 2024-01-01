
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_f = nn.Linear(2, 2)
        self.layers_j = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers_f(x)
        x = self.layers_j(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.cat([x, x], dim=1)
        return x
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
