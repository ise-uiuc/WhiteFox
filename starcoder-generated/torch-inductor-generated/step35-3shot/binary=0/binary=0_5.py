
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=4, dilation=1, groups=2)
    def forward(self, x):
        y = self.conv1(x)
        return y
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)
