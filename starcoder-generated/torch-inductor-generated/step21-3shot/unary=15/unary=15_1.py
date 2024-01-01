
import math
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, stride=2, padding=1)
        f2_channels = self.conv2.out_channels
        f2_width = math.floor(f2_channels/self.conv2.groups)
        self.avgpool = torch.ops.aten.avg_pool2d(self.conv2.weight, f2_width, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = torch.mean(v4, dim=0)
        v6 = v5.reshape(-1)
        v7 = torch.matmul(v6, self.avgpool)
        v8 = v7.relu()
        v9 = v8 + v1 # fusion pattern
        return v9
x1 = torch.randn(1, 3, 1280, 4)
x2 = torch.randn(128, 51, 4, 4)
