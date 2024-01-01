
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 8)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = torch.flatten(x, 1)
        x = torch.stack((x, x), dim=0)
        x = torch.cat(x, dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 3)
