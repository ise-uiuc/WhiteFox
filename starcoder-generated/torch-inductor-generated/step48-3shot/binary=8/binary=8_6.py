
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.softmax1 = torch.nn.Softmax()
        self.softmax2 = torch.nn.Softmax()
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.softmax1(v1)
        v5 = self.softmax2(v2)
        v4 = v3 + v5
        v6 = self.conv3(x1)
        v16 = self.conv4(x2)
        v17 = v6 - v16
        v7 = self.conv3(x1)
        v19 = self.conv4(x2)
        v13 = v7.mul(v19)
        v11 = self.conv3(x1)
        v23 = self.conv4(x2)
        v20 = self.conv3(x1)
        v22 = self.conv4(x2)
        v21 = v20.div(v22)
        v12 = v11.add(v13)
        v9 = self.conv3(x1)
        v26 = self.conv4(x2)
        v14 = v9.mul(v26)
        v28 = self.conv4(x2)
        v27 = v9.add(v28)
        v15 = v14.add(v17)
        v10 = v12.div(v21)
        v8 = v10 + v15
        v24 = self.conv3(x1)
        v25 = self.conv4(x2)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
# Model begins
