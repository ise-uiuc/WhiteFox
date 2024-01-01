
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(4, 3)
        self.layers_2 = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.stack((x, x), dim=1)
        x = self.layers_2(x)
        y = torch.cat((x, x), dim=1)
        return y
# Inputs to the model
x = torch.randn(2, 4)
