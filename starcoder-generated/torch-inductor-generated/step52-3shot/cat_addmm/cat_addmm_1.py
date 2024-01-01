
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x], dim=0)
        x = x + 10
        return x
# Inputs to the model
x = torch.randn(2, 2)
