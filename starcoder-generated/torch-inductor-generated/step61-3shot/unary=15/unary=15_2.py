
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(1, 1, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(1, 1, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv1d(1, 1, 9, stride=1, padding=4)
    def forward(self, x1):
        v1 = x1
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = v5.permute(0, 2, 1)
        v7 = self.conv3(v6)
        v8 = v7.contiguous().permute(0, 2, 1)
        v9 = torch.relu(v8)
        v10 = v9.permute(0, 2, 1)
        v11 = self.conv4(v10)
        v12 = v11.permute(0, 2, 1)
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(2, 1, 1001)
