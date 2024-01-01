
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 768, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(128)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.softmax(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
