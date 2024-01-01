
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (3, 5), stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, (1, 5), stride=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.max_pool2d(v2, kernel_size=5, stride=2, padding=2, dilation=1, ceil_mode=False)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 256)
