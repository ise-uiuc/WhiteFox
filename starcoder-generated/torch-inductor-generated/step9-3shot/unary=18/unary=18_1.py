
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1, torch.sigmoid(x1))
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2, torch.sigmoid(v1))
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 40, 40)
