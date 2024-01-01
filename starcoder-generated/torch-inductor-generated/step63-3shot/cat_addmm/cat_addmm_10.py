
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = x.permute(0, 1, 3, 2)
        x = torch.cat((x, x), dim=2)
        x = torch.stack((x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 3)
