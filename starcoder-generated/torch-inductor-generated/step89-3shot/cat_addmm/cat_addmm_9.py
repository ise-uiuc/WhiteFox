
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layer_1 = nn.Linear(2, 3)
        self.layer_2 = nn.Linear(3, 1)
        self.cat = torch.cat
        self.layers.add_module('layer_1', self.layer_1)
        self.layers.add_module('layer_2', self.layer_2)
    def forward(self, x):
        x = self.layers(x)
        x = self.cat((x, x, x))
        x = self.cat((x, x, x))
        x = self.cat((x, x, x))
        return x
# Inputs to the model
x = torch.randn(2, 2)
