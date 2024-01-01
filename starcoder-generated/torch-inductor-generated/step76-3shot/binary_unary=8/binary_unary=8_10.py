
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1)
        self.conv7 = torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.conv8 = torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(x1)
        v4 = self.conv4(v3)
        v5 = self.conv5(x1)
        v6 = self.conv6(v5)
        v7 = self.conv7(x1)
        v8 = self.conv8(v7)
        v9 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
        v10 = torch.relu(v9)
        v11 = v10.permute(0, 2, 3, 1)
        return v11.matmul(v11)
# Inputs to the model
x1 = torch.randn(1, 4, 100, 100)
