
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x), dim=0)
        x = x.chunk(3, dim=0)[0]
        x = torch.stack((x, x), dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 5)
