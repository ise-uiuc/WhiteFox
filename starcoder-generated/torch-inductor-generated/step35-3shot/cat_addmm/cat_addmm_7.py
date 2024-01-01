
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 16)
        self.layers2 = nn.Linear(16, 32)
        self.layers3 = nn.Linear(32, 32)
        self.layers4 = nn.Linear(64, 64)
    def forward(self, x1, x2, x3, x4):
        x1 = self.layers(x1)
        x2 = self.layers2(x2)
        x3 = self.layers3(x3)
        x4 = self.layers4(x4)
        x3 = torch.cat([x3], dim=2)
        x3 = torch.stack([x3])
        x3 = x3.repeat(2, 3, 1)
        x1 = torch.cat([x1, x2, x3], dim=0)
        x4 = torch.stack([x4])
        x4 = x4.repeat(1,3, 1)
        x5 = torch.cat([x4], dim=0)
        x6 = torch.cat([x5], dim=0)
        return x1, x6
# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1,16)
x3 = torch.randn(1, 32)
x4 = torch.randn(1, 64)
