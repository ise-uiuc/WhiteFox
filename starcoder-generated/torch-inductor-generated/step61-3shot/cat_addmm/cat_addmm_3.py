
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 8)
    def forward(self, x):
        x = self.layers.linear(x)
        x = torch.stack((x, x, x, x), dim=1).flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 6)
