
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 8, stride=2, padding=4)
        self.conv2 = torch.nn.Conv2d(3, 64, 8, stride=2, padding=4)
        self.conv3 = torch.nn.Conv2d(3, 64, 4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 64, 4, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v3 = self.conv3(x2)
        v2 = self.conv2(x1)
        v4 = self.conv4(x2)
        v9 = v1 + v3
        v8 = v2 + v4
        return v9, v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
