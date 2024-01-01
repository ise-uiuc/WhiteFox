
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3,2)
        self.stack = torch.stack
        self.cat = torch.cat

    def forward(self, x):
        x = self.layers(x)
        x = self.stack(x, dim=0)
        x = self.cat(x, x, dim=1)
        x = self.stack(x, dim=1)
        return x
# Inputs:
x = torch.randn(1, 3)
