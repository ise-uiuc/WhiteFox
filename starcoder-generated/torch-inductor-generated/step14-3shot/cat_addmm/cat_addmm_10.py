
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.layers2 = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        out = torch.cat((x, x, x), dim=1)
        out = self.layers2(out)
        out = self.layers(out)
        out = torch.cat((out, out, out), dim=1)
        return out
# Inputs to the model
x = torch.randn(2, 2)
