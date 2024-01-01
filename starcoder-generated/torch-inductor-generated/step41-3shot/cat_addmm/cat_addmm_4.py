
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
        self.cat = torch.cat
    def forward(self, x1):
        x = self.layers(x1)
        x = self.cat([x, x], dim=0)
        return x
# Inputs to the model
x = torch.randn(1)
