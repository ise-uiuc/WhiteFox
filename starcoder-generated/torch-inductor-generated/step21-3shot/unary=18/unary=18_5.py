
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v3 = self.conv1(x1)
        v1 = torch.sigmoid(v3)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
