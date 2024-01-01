
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 6, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.bn2 = torch.nn.BatchNorm2d(6)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = v1.detach()
        v5 = v2.detach()
        v6 = v1 + v2
        v7 = v3 + v4
        v8 = v5 + v7
        return v8
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
