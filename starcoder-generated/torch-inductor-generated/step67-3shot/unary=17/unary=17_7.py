
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 64, 1, stride=2)
        self.depthwise_conv = torch.nn.Conv2d(64, 64, 3, padding=1, groups=64, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        x1 = self.depthwise_conv(x1)
        x1 = self.conv_transpose(x1)
        x1 = self.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 27, 27)
