
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (9, 1), stride=(1, 1), padding=(4, 0))
        self.conv2 = torch.nn.Conv2d(64, 96, (1, 9), stride=(1, 1), padding=(0, 4))
        self.conv3 = torch.nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(96, 72, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5 = torch.nn.Conv1d(72, 64, (1L,), stride=(1L,), padding=(0L,))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
