
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 4)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(1, 4, 1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = Model1()
        self.layers = nn.Sequential(nn.Linear(12, 12))
    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(1)
