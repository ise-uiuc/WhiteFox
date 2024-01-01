
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        y = x + x
        x = y.add(2)
        z = y.add(3)
        x = torch.cat((x, z), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
