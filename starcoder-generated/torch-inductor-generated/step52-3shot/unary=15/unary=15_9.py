
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.010000000000000009, affine=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.010000000000000009, affine=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.norm1(v1)
        v3 = self.relu1(v2)
        v4 = self.conv2(v3)
        v5 = self.norm2(v4)
        v6 = self.relu2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(3, 3, 640, 640)
