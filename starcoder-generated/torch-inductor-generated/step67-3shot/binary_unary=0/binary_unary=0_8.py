
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = torch.cat([x1,x2,x3,x4],axis=0)
        v1 = self.conv1(a1)
        v2 = v1 + x1
        v31 = torch.relu(v2)

        v32 = x1 + x2 + x3
        a2 = torch.cat([v32, v31, x4],axis=0)
        v4 = self.conv2(a2)
        v5 = v4 + v1
        v6 = torch.relu(v5)
        v7 = x4 + v6
        v8 = torch.relu(v7)
        v9 = v7
        v10 = v1 + self.conv3(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(256, 3, 224, 224)
x2 = torch.randn(64, 3, 224, 224)
x3 = torch.randn(64, 3, 224, 224)
x4 = torch.randn(32, 3, 224, 224)
