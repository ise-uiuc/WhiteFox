
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(24, 6)
    def forward(self, x):
        x = torch.stack((x, x), dim=1).flatten(1)
        x = self.layers(x)
        x = x.transpose(1, 0).flatten(1)
        return x
# Inputs to the model
x = torch.randn(8, 2, 3)
