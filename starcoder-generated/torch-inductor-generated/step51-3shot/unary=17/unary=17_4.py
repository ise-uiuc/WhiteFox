
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=True)
        self.conv2 = torch.nn.ConvTranspose2d(32, 3, 2, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 15, 20)
