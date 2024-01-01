
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 11, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 - 0.2
        x3 = F.relu(x3)
        x4 = self.conv(x3)
        x5 = x4 - 0.1
        x6 = F.relu(x5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
