
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.relu2(x)

        return y + y
# Inputs to the model
x = torch.randn(1, 16, 56, 56)
y = torch.randn(1, 16, 56, 56)
