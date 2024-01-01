
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('layers_0', nn.Linear(2, 2))
    def forward(self, x):
        x = self.layers(x)
        for i in range(x.size(-1)):
            torch.flatten(x, start_dim=i + 1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
