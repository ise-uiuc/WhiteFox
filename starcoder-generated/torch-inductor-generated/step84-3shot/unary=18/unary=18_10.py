
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv12 = torch.nn.Conv1d(4, 24, 1, stride=1, padding=0)
        self.conv21 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv22 = torch.nn.Conv1d(4, 24, 1, stride=1, padding=0)
        self.conv31 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv32 = torch.nn.Conv1d(4, 24, 1, stride=1, padding=0)
        self.conv41 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv42 = torch.nn.Conv1d(4, 24, 1, stride=1, padding=0)
        self.conv51 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv52 = torch.nn.Conv1d(4, 24, 1, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(192, 64)
    def forward(self, x1):
        v1 = self.conv11(x1)
        v2 = self.conv12(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv21(v3)
        v5 = self.conv22(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv31(v6)
        v8 = self.conv32(v7)
        v9 = torch.sigmoid(v8)
        v10 = self.conv41(v9)
        v11 = self.conv42(v10)
        v12 = torch.sigmoid(v11)
        v13 = self.conv51(v12)
        v14 = self.conv52(v13)
        v15 = torch.sigmoid(v14)
        v16 = v14 + v9
        v17 = v5 + v8
        v18 = v1 + v7
        v19 = torch.cat([v17, v18, v16, v15], dim=1)
        v20 = torch.reshape(v19, (-1, 192))
        v21 = self.fc1(v20)
        v22 = torch.sigmoid(v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 1, 400)
