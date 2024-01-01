
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        return x
# Inputs to the model
x = torch.randn(8, 2)
