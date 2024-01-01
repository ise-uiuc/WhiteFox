
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 3)
    def forward(self, x):
        x = self.layers(x)
        input = torch.stack((x, x, x, x, x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 8)
