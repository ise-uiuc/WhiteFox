
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity1 = torch.nn.Identity()
        self.identity2 = torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=3, padding=3)
    def forward(self, x):
        t1 = self.identity1(x)
        v1 = self.identity2(t1)
        v2 = self.conv1(v1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 + v4
        v7 = torch.relu(v6)
        v8 = self.conv3(v7)
        v9 = v8 + v4
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
