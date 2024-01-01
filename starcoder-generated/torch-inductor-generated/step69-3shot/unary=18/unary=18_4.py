
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
          nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=0),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block2 = nn.Sequential(
          nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2, stride=1))
        self.conv_block3 = nn.Sequential()
        self.conv_block3.add_module("conv3a", nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0))
        self.conv_block3.add_module("relu3a", nn.ReLU(True))
        self.conv_block3.add_module("conv3b", nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0))
    def forward(self, x1):
        out = self.conv_block1(x1)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        return out
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
