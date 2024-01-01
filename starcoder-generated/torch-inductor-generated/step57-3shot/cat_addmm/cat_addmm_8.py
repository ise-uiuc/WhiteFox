
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=0)
        x = torch.stack((x, x), dim=0)
        x = x[0, 0]
        return x
# Inputs to the model
x = torch.randn(4, 7)
