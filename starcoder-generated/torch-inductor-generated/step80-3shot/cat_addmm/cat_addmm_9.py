
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
        self.layers2 = nn.Linear(1, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x))
        x = x.flatten(1)
        x = self.layers2(x)
        return x
# Inputs to the model
x = torch.randn(2)
