
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 1, 5, stride=2)
        self.conv1 = torch.nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 64, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = v1.sum(dim=1, keepdim=True)
        v3 = self.conv1(v2)
        v4 = self.conv2(v3)
        v5 = v4.sum(dim=1, keepdim=True)
        v6 = F.sigmoid(v5)
        v7 = v4 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
