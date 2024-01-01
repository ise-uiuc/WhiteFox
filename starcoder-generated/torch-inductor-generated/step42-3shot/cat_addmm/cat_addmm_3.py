
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x], dim=0)
        x = self.stack([x, x], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
