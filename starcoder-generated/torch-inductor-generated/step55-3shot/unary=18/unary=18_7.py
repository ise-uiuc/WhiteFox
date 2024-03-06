
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=1)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=1)
        self.pool = torch.nn.AvgPool2d(int(51))
        self.fc = torch.nn.Linear(512, 10)
        self.conv8 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.conv9 = torch.nn.Conv2d(1024, 512, kernel_size=1)
        self.pool2 = torch.nn.AvgPool2d(int(51))
        self.conv10 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.conv11 = torch.nn.Conv2d(1024, 512, kernel_size=1)
        self.conv12 = torch.nn.Conv2d(512, 8, kernel_size=3, padding=1, stride=1)
        self.conv13 = torch.nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)
        self.pool3 = torch.nn.AvgPool2d(int(51))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        v11 = self.conv6(v10)
        v12 = torch.relu(v11)
        v13 = self.conv7(v12)
        v14 = torch.relu(v13)
        v15 = self.pool(v14)
        v16 = v15.flatten()
        v17 = self.fc(v16)
        v18 = torch.relu(v17)
        v19 = self.conv8(v14)
        v20 = self.conv9(v19)
        v21 = torch.softmax(v20, dim=1)
        v22 = self.pool2(v20)
        v23 = self.conv10(v21)
        v24 = self.conv11(v23)
        v25 = torch.sigmoid(v24)
        v26 = self.conv12(v25)
        v27 = self.conv13(v26)
        v28 = torch.sigmoid(v27)
        v29 = self.pool3(v28)
        return v29
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)