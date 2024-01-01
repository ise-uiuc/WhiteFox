
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.stack = torch.stack
    def forward(self, x):
        for index in range(3):
            x = self.layers(x)
        x = self.stack((x, x, x), dim = 1)
        x = x.sum(1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
