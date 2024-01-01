
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 3, 2, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 2, 1, stride=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + v1
        return v4
# Inputs to the model
x = torch.randn(1, 3, 44, 44)
