
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 8)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=0)
        x = torch.flatten(x, 0)
        x = torch.split(x, 2, dim=0)
        return x
# Inputs to the model
x = torch.randn(8, 2)
