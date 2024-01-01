
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 6)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x).pow(2)
        x = self.stack((x, x), dim=1)
        x = x.flatten(1)
        x = x.T.flatten(1)
        return x
# Inputs to the model
x = torch.randn(7, 7)
