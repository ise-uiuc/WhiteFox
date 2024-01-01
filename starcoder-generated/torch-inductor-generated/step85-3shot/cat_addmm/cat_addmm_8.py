
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.stack((x, x, x), dim=2)
        x = x.flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
