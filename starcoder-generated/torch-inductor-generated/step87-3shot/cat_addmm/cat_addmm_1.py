
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 50)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=0)
        x = torch.cat((x, x), dim=0)
        x = x[15]
        return x
# Inputs to the model
x = torch.randn(10, 100)
