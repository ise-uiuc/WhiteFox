
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=0)
        x = torch.cat(["foo", x[0], x[1], "bar"], dim=0)
        return x
# Inputs to the model
x = torch.randn(3, 3)
