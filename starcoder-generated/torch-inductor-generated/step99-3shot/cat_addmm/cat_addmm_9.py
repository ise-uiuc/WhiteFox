
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
    def forward(self, x):
        x = self.cat([x, x, x], dim=1)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 1)
