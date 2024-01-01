
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, dilation=1)
    def forward(self, x):
        x = self.conv3(x)
        x = self.conv1(x)
        x = x - 4.2e-05
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
