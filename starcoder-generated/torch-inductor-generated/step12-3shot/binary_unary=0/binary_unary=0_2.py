
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 13, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = x3 * v1
        v3 = self.conv1(x2)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1=torch.randn(1, 1, 15 * 13, 15 * 13)
x2=torch.randn(1, 16, 64, 64)
x3=torch.randn(1, 1, 13, 13)
