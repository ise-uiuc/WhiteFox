
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 10, stride=5, padding=5)
        self.conv2 = torch.nn.Conv2d(8, 16, 10, stride=5, padding=5)
        self.conv3 = torch.nn.Conv2d(16, 32, 10, stride=5, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 1
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
