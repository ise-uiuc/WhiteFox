
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.layers(x)
        y = self.flatten(x)
        y = y + y
        return y
# Inputs to the model
x = torch.randn(2, 2)
