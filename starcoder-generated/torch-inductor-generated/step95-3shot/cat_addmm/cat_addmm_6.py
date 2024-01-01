
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, torch.cat((x, x), dim=1), torch.cat((x, x), dim=1)), dim=0).flatten(1).flatten(1)
        return x
# Inputs to the model
x = torch.randn(4, 2)
