
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(bias=True)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, groups=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
