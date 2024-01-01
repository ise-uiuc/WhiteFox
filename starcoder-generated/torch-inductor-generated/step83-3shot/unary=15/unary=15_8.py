
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = torch.nn.functional.interpolate(v2, size=[17, 17], mode='nearest')
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = torch.nn.functional.interpolate(v5, size=[33, 33], mode='nearest')
        v7 = self.conv3(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 200, 200)
