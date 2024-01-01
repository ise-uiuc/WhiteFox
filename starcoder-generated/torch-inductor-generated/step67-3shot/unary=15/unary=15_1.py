
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(320, 600, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(600, 320, 3, stride=1, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv1d(320, 160, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv1d(160, 160, 2, stride=2, dilation=2)
        self.conv5 = torch.nn.Conv1d(160, 48, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv1d(48, 48, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv1d(48, 18, 3, stride=1, padding=1)
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
        return v14
# Inputs to the model
x1 = torch.randn(1, 500, 320)
