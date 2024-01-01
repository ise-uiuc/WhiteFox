
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten())
        self.layers.add_module('flatten', nn.Flatten())
        self.layers.add_module('flatten2', torch.flatten)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 2)
