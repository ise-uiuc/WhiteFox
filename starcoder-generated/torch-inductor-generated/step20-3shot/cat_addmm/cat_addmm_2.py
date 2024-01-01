
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.layers0 = nn.Linear(4, 9)
        self.layers1 = nn.Linear(9, 7)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x), dim=1)
        x = self.layers0(x)
        x = torch.cat((x, x, x), dim=1)
        x = self.layers1(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
