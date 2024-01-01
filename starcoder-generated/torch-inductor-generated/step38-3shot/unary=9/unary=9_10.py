
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 5, stride=2, padding=(2, 0), dilation=2, bias=True)
        self.conv2 = torch.nn.Conv2d(8 * 2, 16, 1, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(4, 16, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = torch.cat((v2, v3), dim=1)
        v5 = torch.nn.functional.relu6(v4)
        v6 = 3 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64) # shape of input image is 64x64
