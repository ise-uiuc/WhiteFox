
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 4)
        self.layers2 = nn.Linear(3, 4)
    def forward(self, x):
        x = self.layers(x)
        x = x.expand(2, 5, 4)
        x = self.layers2(x)
        x = torch.cat((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(3, 5)
