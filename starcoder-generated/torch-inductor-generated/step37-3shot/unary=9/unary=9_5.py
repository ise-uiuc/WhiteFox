
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(12, 1, 1)
    def forward(self, x1):
        x1_conv1 = self.conv1(x1)
        x1_conv2 = self.conv2(x1_conv1)
        x1_conv1_resize = torch.nn.functional.interpolate(x1_conv2, scale_factor=2)
        x = torch.cat([x1_conv1_resize, x1_conv1], dim=1)
        x1_conv3 = self.conv3(x)
        return x1_conv3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
