
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.cat([x, x], dim=0)
        x = self.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn((2, 4))
