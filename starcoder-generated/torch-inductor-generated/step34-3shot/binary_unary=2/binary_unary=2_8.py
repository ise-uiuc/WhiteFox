
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (11, 11), stride=(1, 1), padding=(5, 5))
        self.conv2 = torch.nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(32, 48, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv4 = torch.nn.Conv2d(1, 3, (5, 5), stride=(1, 1), padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        v5 = self.conv2(v4)
        v7 = v5 - 0.1
        v6 = F.relu(v7)
        v8 = self.conv3(v6)
        v10 = v8 - 0.1
        v9 = F.relu(v10)
        v11 = torch.squeeze(v9, 0)
        v13 = self.conv4(v11)
        v15 = v13 - 0.1
        v16 = F.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 23, 23)
