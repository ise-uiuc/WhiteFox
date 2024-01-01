
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1)
        self.conv4 = torch.nn.Conv3d(in_channels = 16, out_channels=64, kernel_size=(7, 7, 7), stride=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1)) * torch.sigmoid(self.conv2(x1))
        v2 = self.conv3(v1)
        v3 = self.conv4(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
