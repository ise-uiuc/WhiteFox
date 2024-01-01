
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.bn = nn.BatchNorm1d(4)
    def forward(self, x):
        x = self.layers(x)
        x = self.bn(x)
        torch.cat((x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
# model ends

# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.lrelu(x)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
# model ends

# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.prelu(x)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
# model ends
