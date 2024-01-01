
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(56, 9, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv2(x2)
        v3 = v1 - v2
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 56, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
