
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 7, stride=1, padding=0, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 8, 1, stride=1, padding=0, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.norm1(v1)
        v3 = torch.mul(v1, x2)
        v4 = v2 + v3
        v5 = self.conv2(v4)
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
x2 = torch.randn(1, 6, 56, 56)
