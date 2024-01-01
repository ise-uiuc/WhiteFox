
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=2)
        self.conv7 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=3)
        self.conv8 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=3)
        self.conv9 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, dilation=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + x1
        v4 = torch.relu(v3)
        v5 = v4 + v1
        v6 = torch.relu(v5)
        v7 = v2 + v6
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = self.conv4(v8)
        v11 = v9 + v8
        v12 = torch.relu(v11)
        v13 = self.conv5(v12)
        v14 = self.conv6(v12)
        v15 = v13 + v12
        v16 = torch.relu(v15)
        v17 = self.conv7(v12)
        v18 = self.conv8(v12)
        v19 = v17 + v18
        v20 = torch.relu(v19)
        v21 = self.conv9(x1)
        v22 = self.conv10(x1)
        v23 = v11 + v22
        v24 = torch.tanh(v23)
        v25 = self.conv7(v24)
        v26 = v25 + v24
        v27 = torch.relu(v26)
        return v27
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
