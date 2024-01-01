
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(7, stride = 1, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(7, stride = 1, padding=0)
        self.maxpool3 = torch.nn.MaxPool2d(7, stride = 1, padding=0)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        a1 = torch.sin(v2)
        v3 = self.conv2(x2)
        v4 = torch.sin(v3)
        v5 = v2 + v4
        a2 = a1 + v5
        v6 = torch.relu(a2)
        v7 = self.conv3(x3)
        a3 = torch.sigmoid(v2)
        v8 = torch.tanh(v7)
        a4 = v8 + v7
        v9 = a4 + torch.sinh(v7)
        v10 = torch.relu(v9)
        v11 = self.maxpool1(v1)
        a5 = torch.sigmoid(v11)
        v12 = torch.exp(a5)
        v13 = self.maxpool2(v12)
        a6 = torch.sin(v10)
        v14 = self.maxpool3(a6)
        a7 = a3 + v14
        v15 = torch.relu(a7)
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.tanh(torch.randn(1, 16, 64, 64))
x4 = torch.sigmoid(torch.randn(1, 16, 64, 64))
