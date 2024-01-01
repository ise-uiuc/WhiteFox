
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v = self.conv1(x)
        v1 = self.conv2(v)
        v2 = v1 * x
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
