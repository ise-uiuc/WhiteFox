 (22 input channels)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(22, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model (22 input channels) (3 color channels + 12 edge maps from U-Net decoder side)
x1= torch.randn(1, 22, 64, 64)
