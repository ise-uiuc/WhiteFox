
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
        self.layers1 = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1).flatten(start_dim=1)
        return self.layers1(x)
# Inputs to the model
x = torch.randn(2, 2)
