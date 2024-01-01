
with torch.no_grad():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Linear(3, 5)
        def forward(self, x):
            return self.layers(x)

class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
    def forward(self, x):
        x = self.model(x)
        x = torch.stack((x, x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 3)
