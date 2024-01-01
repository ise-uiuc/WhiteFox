
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v0 = v1 - v1
        v2 = F.relu(v0)
        v3 = self.conv2(v2)
        v4 = v2 - v2
        v5 = F.relu(v4)
        v7 = torch.squeeze(v5, 0)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
