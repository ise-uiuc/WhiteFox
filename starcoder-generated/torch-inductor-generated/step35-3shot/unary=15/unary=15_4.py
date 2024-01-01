
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 3, 3, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        x2 = self.conv2(v2)
        x3 = self.conv3(x2)
        x4 = torch.relu(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
