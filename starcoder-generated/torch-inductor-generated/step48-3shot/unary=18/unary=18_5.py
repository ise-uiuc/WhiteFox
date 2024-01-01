
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=23, kernel_size=4, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
