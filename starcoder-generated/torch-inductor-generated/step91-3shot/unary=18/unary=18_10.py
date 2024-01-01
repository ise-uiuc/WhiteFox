
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(4, stride=4, padding=0)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.pool(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        return torch.cat([v1, v5], dim=1)
# Inputs to the model
x = torch.randn(1, 1, 256, 256)
