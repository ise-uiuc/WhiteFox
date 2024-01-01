
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        z = torch.stack((x,), dim=1)
        y = torch.cat((z, x), dim=1)
        return y
# Inputs to the model
x = torch.randn(2, 3)
