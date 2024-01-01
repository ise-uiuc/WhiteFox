
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.conv2_drop = torch.nn.Dropout(0.5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.softmax(v2, dim=1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
