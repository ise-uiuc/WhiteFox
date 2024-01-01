
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.add = torch.addmm
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.add(x, x, x)
        x = self.cat((x, x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
