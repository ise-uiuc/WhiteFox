
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t1 = 5 * v1
        v2 = self.conv2(x1)
        t2 = 2 * t1 * v2
        t3 = t2 + 5 - 2 / 3
        t4 = t1 * t2 * t3
        v3 = torch.relu(5 * t1 + t2 - t3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 16)
