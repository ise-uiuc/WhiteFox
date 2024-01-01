
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 100, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv1d(100, 800, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(800, 4800, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(4800, 800, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(800, 400, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv1d(400, 800, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv1d(800, 800, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv1d(800, 800, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv1d(800, 800, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv1d(800, 800, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv1d(800, 1600, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv1d(1600, 1600, 1, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = self.conv6(v6)
        v8 = self.conv7(v7)
        v9 = self.conv8(v8)
        v10 = self.conv9(v9)
        v11 = self.conv10(v10)
        v12 = self.conv11(v11)
        v13 = v12 * 0.5
        v14 = v12 * v12
        v15 = v14 * v12
        v16 = v15 * 0.044715
        v17 = v12 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v13 * v20
        return v21
# Inputs to the model
x2 = torch.randn(1, 1, 15)
