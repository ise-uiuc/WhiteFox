
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1) + self.conv3(v1)
        v3 = torch.relu(v2 + self.conv4(v1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
