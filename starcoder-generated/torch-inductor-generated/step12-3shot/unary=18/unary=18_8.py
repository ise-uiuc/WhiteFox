
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv1(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
