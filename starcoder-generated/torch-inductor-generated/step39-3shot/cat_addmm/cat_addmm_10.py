
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 100)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.reshape(4, 100)
        return x
# Inputs to the model
x = torch.randn(8, 100)
