
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.concat((x, x), dim=3)
        x = torch.sum(x, dim=3).view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
