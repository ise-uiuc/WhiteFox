
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)        
    def forward(self, x1):
        v0 = self.conv1(x1)
        v1 = v0 - 1
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 - 1
        v5 = F.relu(v4)
        v6 = self.conv3(v5)
        v7 = v6 - 1
        v8 = F.relu(v7)
        v9 = self.conv4(v8)
        v10 = v9 - 1
        v11 = F.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
