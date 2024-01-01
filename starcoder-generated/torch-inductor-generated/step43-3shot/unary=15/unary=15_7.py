
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, stride=1, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv2d(16, 64, 3, stride=1, padding=1, dilation=4)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, dilation=8)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        v1 = torch.relu(y4)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
