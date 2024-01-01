
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=3)
    def forward(self, x, x2, x3):
        x1 = self.conv1(x)
        x4 = self.conv2(x1)
        v1 = x4 + x3
        v2 = torch.relu(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        v5 = x2 + v1
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
