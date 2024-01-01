
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv11 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, *argv):
        v1 = self.conv1(*argv)
        v2 = self.conv1(*argv)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv2(*argv)
        v6 = self.conv2(*argv)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v9 = self.conv3(*argv)
        v10 = self.conv3(*argv)
        v11 = self.conv3(*argv)
        v12 = v9 + v10 + v11
        v13 = torch.relu(v12)
        v14 = self.conv4(*argv)
        v15 = self.conv4(*argv)
        v16 = self.conv4(*argv)
        v17 = self.conv4(*argv)
        v18 = v14 + v15 + v16 + v17
        v19 = torch.relu(v18)
        v20 = self.conv5(*argv)
        v21 = self.conv5(*argv)
        v22 = v14 + v15 + v16 + v17
        v23 = torch.relu(v22)
        v24 = self.conv6(*argv)
        v25 = self.conv6(*argv)
        v26 = v24 + v25
        v27 = torch.relu(v26)
        v28 = self.conv7(*argv)
        v29 = self.conv7(*argv)
        v30 = v28 + v29
        v31 = torch.relu(v30)
        v32 = self.conv8(*argv)
        v33 = self.conv8(*argv)
        v34 = v32 + v33
        v35 = torch.relu(v34)
        v36 = self.conv9(*argv)
        v37 = self.conv9(*argv)
        v38 = v36 + v37
        v39 = torch.relu(v38)
        v40 = self.conv10(*argv)
        v41 = self.conv10(*argv)
        v42 = v40 + v41
        v43 = torch.relu(v42)
        v44 = v4 + v8 + v12 + v18 + v23 + v27 + v31 + v35 + v39 + v43
        return v44
# Inputs to the model
x1 = torch.randn(1, 3, 65, 33)
x2 = torch.randn(1, 3, 65, 33)
x3 = torch.randn(1, 3, 65, 33)
x4 = torch.randn(1, 3, 65, 33)
