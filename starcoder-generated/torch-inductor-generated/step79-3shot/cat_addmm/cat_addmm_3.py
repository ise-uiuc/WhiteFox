
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
        self.cat = torch.cat
    def forward(self, x, y):
        x = self.layers(x)
        y = self.layers(y)
        o = torch.cat((x, y), dim=0)
        o = self.cat((o, o), dim=-1)
        return o
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
