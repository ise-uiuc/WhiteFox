
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 12, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(12)
        self.conv3 = torch.nn.Conv2d(12, 12, 3, stride=2, padding=1)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        return self.conv3(v4)
# Inputs to the model
inputs = torch.randn(2, 12, 224, 224)
