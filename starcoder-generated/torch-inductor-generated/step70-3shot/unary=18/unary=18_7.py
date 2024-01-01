
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.conv6(v10)
        v12 = torch.sigmoid(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
