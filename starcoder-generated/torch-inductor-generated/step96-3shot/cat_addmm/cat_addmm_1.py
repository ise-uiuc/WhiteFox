
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(7, 7)
        self.layers_2 = nn.Linear(7, 7)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = self.cat((x, x), dim=2)
        x = torch.cat((x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(3, 2, 7)
