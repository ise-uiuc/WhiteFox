
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 3, 7, padding=4)
        self.conv4 = torch.nn.Conv2d(3, 3, 9, padding=8)
        self.conv5 = torch.nn.Conv2d(3, 3, 11, padding=16)
        self.conv6 = torch.nn.Conv2d(3, 3, 11, padding=12)
        self.conv7 = torch.nn.Conv2d(3, 3, 11, padding=18)
        self.conv8 = torch.nn.Conv2d(3, 3, 15, stride=2, padding=3)
        self.conv9 = torch.nn.Conv2d(3, 3, 13, stride=3, padding=5)
        self.conv10 = torch.nn.Conv2d(3, 3, 11, stride=5, padding=7)
        self.conv11 = torch.nn.Conv2d(3, 3, 15, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v12 = self.conv2(x)
        v2 = v1 + v12
        v4 = torch.tanh(v2)
        v5 = self.conv3(x)
        v6 = v4 - v5
        v7 = torch.relu(v6)
        v8 = self.conv4(x)
        v9 = v7 * v8
        v10 = torch.tanh(v9)
        v13 = self.conv5(x)
        v14 = v10 * v13
        v15 = torch.relu(v14)
        v16 = self.conv6(x)
        v17 = v15 * v16
        v18 = torch.sigmoid(v17)
        v19 = self.conv7(x)
        v20 = v18 * v19
        v21 = torch.tanh(v20)
        v22 = self.conv8(x)
        v23 = v21 * v22
        v24 = torch.sigmoid(v23)
        v25 = self.conv9(x)
        v26 = v24 * v25
        v27 = torch.tanh(v26)
        v28 = self.conv10(x)
        v29 = v27 * v28
        v30 = torch.sigmoid(v29)
        v31 = self.conv11(x)
        v32 = v30 * v31
        v33 = torch.relu(v32)
        return v33
# Inputs to the model
x = torch.randn(1, 3, 192, 192)

