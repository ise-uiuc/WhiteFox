
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 19, 7, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(19, 19, 3, stride=1, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv2d(19, 79, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(79, 1, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
# Model begins


