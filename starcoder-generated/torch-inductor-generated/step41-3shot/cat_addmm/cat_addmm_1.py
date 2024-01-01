
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = self.stack((x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 3)
