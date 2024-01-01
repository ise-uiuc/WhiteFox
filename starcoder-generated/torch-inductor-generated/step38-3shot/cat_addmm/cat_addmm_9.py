
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 6)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.flatten(x, 1)
        x = torch.stack((x, x, x), dim=0)
        return x.transpose(0, 1)
# Inputs to the model
x = torch.randn(4, 4)
