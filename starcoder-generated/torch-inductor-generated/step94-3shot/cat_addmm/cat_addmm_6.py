
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.stack = torch.stack
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.stack((x, x), dim=1).flatten(1)
        x = self.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
