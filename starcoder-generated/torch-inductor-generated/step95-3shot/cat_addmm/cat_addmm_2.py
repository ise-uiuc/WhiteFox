
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
        self.concat = nn.Linear(2, 2)
    def forward(self, x):
        x = x[::3, :]
        x = self.layers(x)
        x = self.concat(x).flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(25, 4)
