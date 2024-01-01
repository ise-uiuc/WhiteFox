
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 - 1.3
        x4 = F.relu(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
