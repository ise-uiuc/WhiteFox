
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv13 = torch.nn.Conv2d(16, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.nn.functional.pad(x1, (2, 2, 3, 3), 'constant', 0)
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        v9 = torch.relu(v8)
        v10 = self.conv5(v9)
        v11 = torch.relu(v10)
        v12 = self.conv6(v11)
        v13 = torch.relu(v12)
        v14 = self.conv7(v13)
        v15 = torch.relu(v14)
        v16 = self.conv8(v15)
        v17 = torch.relu(v16)
        v18 = self.conv9(v17)
        v19 = torch.relu(v18)
        v20 = self.conv10(v19)
        v21 = torch.relu(v20)
        v22 = self.conv11(v21)
        v23 = torch.relu(v22)
        v24 = self.conv12(v23)
        v25 = torch.relu(v24)
        v26 = self.conv13(v25)
        v27 = torch.nn.functional.pad(v26, (1, 1, 2, 2), 'constant', 0)
        return v27
# Inputs to the model
x1 = torch.randn(1, 1, 15, 15)
