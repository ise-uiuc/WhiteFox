
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 2, stride=2, padding=1)
        self.globalpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 10)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.tanh(self.conv2(v1))
        v3 = torch.sigmoid(self.conv3(v2))
        v4 = torch.max(v3, 1)
        v5 = v4.values
        v6 = torch.mul(x1, v4)
        v7 = self.globalpool(v6)
        v8 = v7.view(x1.shape[0], -1)
        v9 = self.fc(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
