
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
        self.layers_3 = nn.Linear(2, 2)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        y = self.layers_2(x)
        a = self.layers_3(x)
        x = self.stack((x, y, a, x, y), dim = 1)
        x = x.mean(1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
