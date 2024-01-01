
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=48, out_channels=8, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 24, 72, 72)
