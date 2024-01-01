
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=0, groups=128, bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=0, bias=False)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=0, groups=128, bias=False)
        self.conv5 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
