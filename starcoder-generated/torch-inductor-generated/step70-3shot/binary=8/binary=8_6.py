
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.tanh(v1)
        v4 = torch.tanh(v2)
        r1 = v3 + v4
        v5 = r1 + v2
        v6 = v1 + v5
        v7 = self.conv3(v6)
        v8 = self.conv4(v5)
        v9 = torch.tanh(v6)
        v10 = torch.tanh(v7)
        r4 = v9 + v10
        v11 = r4 + v5
        v12 = v8 + v11
        v13 = self.conv5(v12)
        v14 = torch.tanh(v13)
        v15 = v6.add(v15)
        v16 = v14.add(v14)
        v17 = v16.add(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
