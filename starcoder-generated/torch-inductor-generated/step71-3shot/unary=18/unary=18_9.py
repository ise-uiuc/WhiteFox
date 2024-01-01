
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=96, out_channels=208, kernel_size=1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=208, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0)
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
x1 = torch.randn(1, 3, 112, 112)
