
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 5)
    def forward(self, x):
        x = torch.addmm(x, self.layers, self.layers)
        x = torch.cat((x, x), dim=1)
        x = torch.stack((x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
