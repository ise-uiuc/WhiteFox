
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(12, 20, 3, stride=1, padding=1, dilation=1, groups=4, bias=False)
        self.conv2d_1 = torch.nn.Conv2d(5, 1, 1, stride=1, dilation=1, padding=0)
        self.conv2d_2 = torch.nn.Conv2d(9, 1, 1, stride=1, dilation=1, padding=0)
        self.conv2d_3 = torch.nn.Conv2d(7, 1, 1, stride=1, dilation=1, padding=0)
        self.conv2d_4 = torch.nn.Conv2d(11, 1, 1, stride=1, dilation=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        t1 = self.conv2d_1(self.conv2d(x1))
        t2 = self.conv2d_2(self.conv2d(x1))
        t3 = self.conv2d_3(self.conv2d(x1))
        t4 = self.conv2d_4(self.conv2d(x1))
        v1 = t1 + t2 + t3 + t4
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
