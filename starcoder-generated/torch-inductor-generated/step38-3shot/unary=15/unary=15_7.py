
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return torch.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
