
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 8)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1).flatten(1).flatten(0)
        return x
# Inputs to the model
x = torch.randn(2, 8)
