
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('layers_0', nn.Linear(2, 2))
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x))
        x = self.cat((x))
        return x
# Inputs to the model
x = torch.randn(2, 2)
