
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 6)
    def forward(self, x):
        x = self.layers(x)
        x = 2.5 * x + 14
        x = torch.stack((x, x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 5)
