
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = torch.nn.functional.avg_pool2d(v1 + v2, 3, stride=2, padding=1)
        v4 = self.conv3(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 8, 8, 8)
