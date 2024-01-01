
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 6, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 8, 5, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + v3
        v5 = F.relu(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
