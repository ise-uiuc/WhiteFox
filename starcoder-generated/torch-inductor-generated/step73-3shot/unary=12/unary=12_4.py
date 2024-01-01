
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, (3, 3), stride=(2, 2), padding=(0, 0), dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(512, 512, (2, 2), stride=(1, 1), padding=(1, 1), dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v3 * v2
        return v6
# Inputs to the model
x1 = torch.randn(1, 512, 64, 64)
