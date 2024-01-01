
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, 1)
        self.conv2 = torch.nn.Conv2d(512, 512, 1)
        self.conv3 = torch.nn.Conv2d(512, 6, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.0
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.0
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
