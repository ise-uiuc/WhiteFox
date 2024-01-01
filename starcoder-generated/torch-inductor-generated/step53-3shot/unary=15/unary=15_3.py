
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, dilation=2, bias=False)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=8, bias=False)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=5, bias=False)
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 192, 192)
