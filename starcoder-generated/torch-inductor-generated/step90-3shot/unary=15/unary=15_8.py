
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=11, padding=7)
        self.conv2 = torch.nn.Conv2d(8, 12, 3, stride=11, padding=7)
        self.conv3 = torch.nn.Conv2d(12, 16, 3, stride=11, padding=7)
        self.conv4 = torch.nn.Conv2d(16, 20, 3, stride=11, padding=7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v0 = torch.nn.functional.max_pool2d(v4, kernel_size=[2, 2], stride=2, padding=0, ceil_mode=False)
        v5 = self.conv3(v0)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = torch.nn.functional.max_pool2d(v8, kernel_size=[3, 3], stride=3, padding=0, ceil_mode=False)
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)
