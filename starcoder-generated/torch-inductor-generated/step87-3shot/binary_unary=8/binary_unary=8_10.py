
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, groups=8, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, groups=8, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
