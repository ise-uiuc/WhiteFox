
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=1, padding=1)
        self.pool2 = torch.nn.AvgPool2d(2)
        self.conv3 = torch.nn.Conv2d(32, 64, 7, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(self.pool1(self.conv1(x1)), None, 5, 'linear')
        v2 = torch.nn.functional.interpolate(self.pool2(self.conv2(v1)), None, 2, 'nearest')
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 320, 256)
