
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.conv1(x1))
        v2 = torch.nn.functional.max_pool2d(v1, 3, stride=2, padding=1, dilation=1, ceil_mode=False)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
