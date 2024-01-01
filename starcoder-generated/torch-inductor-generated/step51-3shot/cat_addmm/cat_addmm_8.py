
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 11)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x[...]
        return x
# Inputs to the model
x = torch.randn(2, 2)
