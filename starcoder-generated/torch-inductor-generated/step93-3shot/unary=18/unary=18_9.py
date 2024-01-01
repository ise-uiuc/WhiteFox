
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 15, (3, 3), stride=(3, 2), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(15, 8, (1, 1), stride=(1, 1), padding=(0, 0))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((20, 20))
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.conv2(v1)
        v3 = self.avgpool(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Input to the model
x1 = torch.randn(1, 3, 15, 15)
