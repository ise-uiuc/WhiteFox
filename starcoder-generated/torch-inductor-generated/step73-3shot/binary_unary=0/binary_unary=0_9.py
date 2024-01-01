
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv7 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv8 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv9 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv10 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv11 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv12 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv13 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv14 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv15 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
        self.conv16 = torch.nn.Conv1d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 + v2
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = self.conv2(v6)
        v8 = v7 + v5
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 + v6
        v12 = torch.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 + v8
        v15 = torch.relu(v14)
        v16 = self.conv6(v15)
        v17 = self.conv2(v16) + v9
        v18 = torch.relu(v17)
        v19 = self.conv7(v18)
        v20 = self.conv2(self.conv1(x1)) + v2
        v21 = torch.relu(v20)
        v22 = self.conv8(v21)
        v23 = self.conv9(v22) + v17
        v24 = torch.relu(v23)
        v25 = self.conv10(v24)
        v26 = v25 + v11
        v27 = torch.relu(v26)
        v28 = self.conv11(v27)
        v29 = self.conv2(self.conv1(v21)) + v28
        v30 = torch.relu(v29)
        v31 = self.conv12(v30)
        v32 = self.conv13(v31) + v23
        v33 = torch.relu(v32)
        v34 = self.conv14(v33)
        v35 = self.conv15(v34) + v18
        v36 = torch.relu(v35)
        v37 = self.conv16(v36)
        v38 = v37 + v15
        v39 = torch.relu(v38)
        return v19
# Inputs to the model
x1 = torch.randn(1, 1, 128)
