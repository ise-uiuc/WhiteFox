
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x, x), dim=1)
        x = self.flatten(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
