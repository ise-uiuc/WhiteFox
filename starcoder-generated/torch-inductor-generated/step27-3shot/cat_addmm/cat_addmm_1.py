
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 10)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=0)
        y = torch.chunk(x, 2, dim=0)
        z = torch.cat((y[0], y[1]), dim=1)
        return z
# Inputs to the model
x = torch.randn(1, 4)
