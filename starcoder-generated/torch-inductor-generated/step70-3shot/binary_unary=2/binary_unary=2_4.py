
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=4, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v2)
        v4 = v1 + v3
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
