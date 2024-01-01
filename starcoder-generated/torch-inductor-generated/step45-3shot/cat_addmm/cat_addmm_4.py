
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 5)
        self.cat = torch.cat
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = self.stack([x, x, x], dim=1)
        x = x.transpose(0, 1)
        x = x.view(-1)
        x = self.cat([x, x, x], dim=1)
        x = self.stack([x, x, x], dim=1)
        x, _ = torch.max(x, dim=0)
        return x
# Inputs to the model
x = torch.randn(5, 5)
