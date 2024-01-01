
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = torch.add(a1, 3)
        a3 = torch.relu(a2)
        a3v = torch.max(a3, 1)
        a3w = torch.min(a3, 1)
        a3v2 = a3v - 100
        a3w2 = a3w + 100
        a3v3 = torch.max(a3v2, a3v2)
        a3w3 = torch.min(a3w2, a3w2)
        a4 = torch.mul(a3w3, 2)
        a5 = torch.relu(a4)
        a6 = self.conv(a5)
        a7 = torch.add(a6, 3)
        a8 = torch.relu(a7)
        return a8
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
