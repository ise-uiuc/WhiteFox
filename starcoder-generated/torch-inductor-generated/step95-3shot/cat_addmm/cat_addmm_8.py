
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 8)
    def forward(self, x):
        x = self.layers(x)
        x = x.narrow(start=1, length=3, dim=1)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 8)
