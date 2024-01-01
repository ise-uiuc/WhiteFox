
class Model(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.relu1 = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU6()
        self.conv1 = torch.nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.conv2 = torch.nn.Conv2d(channels[1], channels[2], kernel_size=1)
        self.conv3 = torch.nn.Conv2d(channels[2], channels[3], kernel_size=1)
        self.conv4 = torch.nn.Conv2d(channels[3], channels[4], kernel_size=1)
        self.conv5 = torch.nn.Conv2d(channels[4], channels[4], kernel_size=1)
        self.conv6 = torch.nn.Conv2d(channels[4], channels[4], kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(channels[1])
        self.bn2 = torch.nn.BatchNorm2d(channels[2])
        self.bn3 = torch.nn.BatchNorm2d(channels[3])
        self.bn4 = torch.nn.BatchNorm2d(channels[4])
        self.bn5 = torch.nn.BatchNorm2d(channels[4])
        self.bn6 = torch.nn.BatchNorm2d(channels[4])

    def forward(self, x):
        v1 = self.relu(x)
        v2 = self.conv1(v1)
        v3 = self.bn1(v2)
        v4 = self.relu1(v3)
        v5 = self.conv2(v4)
        v6 = self.bn2(v5)
        v7 = self.relu1(v6)
        v8 = self.conv3(v7)
        v9 = self.relu6(v8)
        v10 = self.bn3(v9)
        v11 = self.conv4(v10)
        v12 = self.bn4(v11)
        v13 = self.relu1(v12)
        v14 = self.conv5(v13)
        v15 = self.bn5(v14)
        v16 = self.relu1(v15)
        v17 = self.conv6(v16)
        v18 = self.bn6(v17)
        v19 = v18 + v17
        v20 = self.relu6(v19)
        v21 = self.conv6(v20)
        v22 = self.bn6(v21)
        v23 = v22 + v21
        v24 = self.bn2(v23)
        v25 = self.conv3(v24)
        v26 = self.bn3(v25)
        v27 = v26 + v25
        v28 = self.bn1(v27)
        v29 = self.conv2(v28)
        v30 = self.bn2(v29)
        v31 = v30 + v29
        out = self.relu(v31)
        return out
model = Model([3, 16, 16, 24, 24, 96])
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
