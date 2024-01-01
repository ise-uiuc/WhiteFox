
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        v6 = self.conv3(x1)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v10 = self.conv4(v8)
        v11 = v10 + x2
        v12 = torch.relu(v11)
        v14 = self.conv4(v12)
        v15 = v14 + x3
        v16 = torch.relu(v15)
        v18 = self.conv4(v16)
        v19 = v18 + x4
        v20 = torch.relu(v19)
        return v20
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64, requires_grad=True)
x3 = torch.randn(1, 16, 64, 64, requires_grad=True)
x4 = torch.randn(1, 16, 64, 64, requires_grad=True)
