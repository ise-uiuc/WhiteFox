
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v4 = self.conv2(v2)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
