
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
    def forward(self, x1):
        v1 = self.maxpool1(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.conv4(v7)
        v9 = torch.sigmoid(v8)
        v10 = self.conv5(v9)
        v11 = torch.sigmoid(v10)
        v12 = self.conv6(v11)
        v13 = torch.sigmoid(v12)
        v14 = self.conv7(v13)
        v15 = torch.sigmoid(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
