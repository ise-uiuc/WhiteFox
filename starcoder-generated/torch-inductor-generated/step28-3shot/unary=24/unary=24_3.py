
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, stride=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v1)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 224, 224)
