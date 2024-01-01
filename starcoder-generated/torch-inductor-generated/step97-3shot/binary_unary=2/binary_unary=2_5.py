
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 12, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 6, 1, stride=2, padding=0, dilation=2)
        self.conv3 = torch.nn.Conv2d(12, 3, 1, stride=1, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = F.softmax(v3)
        v5 = v4 - 1.0
        v6 = F.celu(v5)
        v7 = F.gelu(v6)
        v8 = self.conv2(v7)
        v9 = v8 - 0.5
        v10 = F.gelu(v9)
        v11 = self.conv3(v10)
        v12 = v11 - 0.5
        v13 = F.relu6(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
