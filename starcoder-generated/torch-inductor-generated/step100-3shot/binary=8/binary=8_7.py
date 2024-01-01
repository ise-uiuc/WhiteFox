
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 9, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(9, 12, 3, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
