. The pattern should appear in the loop.
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=3, dilation=1, groups=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
