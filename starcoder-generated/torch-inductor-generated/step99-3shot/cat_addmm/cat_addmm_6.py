
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=0)
        x = torch.stack((x, x, x), dim=(0, 1))
        y = torch.cat((x, x), dim=1)
        return y
# Inputs to the model
x = torch.randn(1, 2)
