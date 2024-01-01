
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv1d(3, 2, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv1d(4, 6, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(5, 6, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(2, 2, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv1d(6, 6, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv1d(1, 1, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv1d(8, 8, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv1d(7, 4, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv1d(4, 7, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.7343978
        v1 = self.conv0(x)
        v2 = self.conv1(x)
        v3 = torch.add(v1, 1, v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(x)
        v6 = self.conv4(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        v9 = v8 + 1
        v10 = self.conv7(v9)
        v11 = self.conv8(v10)
        v12 = v11 > 0
        v13 = v11 * negative_slope
        v14 = torch.where(v12, v11, v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 8, 5)
