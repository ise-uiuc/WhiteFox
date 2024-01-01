
import collections
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.modules = collections.OrderedDict()
        self.modules['conv1'] = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.modules['conv2'] = torch.nn.Conv2d(4, 4, 1, stride=1)
        self.sequential_1 = torch.nn.Sequential(self.modules)
        self.max_pool2d_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.modules_1 = collections.OrderedDict()
        self.modules_1['conv'] = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.modules_1['leakyrelu'] = torch.nn.LeakyReLU(negative_slope=0.01)
        self.sequential_2 = torch.nn.Sequential(self.modules_1)
    def forward(self, x1):
        v1 = self.sequential_1(x1)
        v2 = self.max_pool2d_1(v1)
        v3 = self.sequential_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 183, 183)
