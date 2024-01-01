
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v4_pool = self.pool(v4, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        v5 = self.conv3(v4_pool)
        v6 = self.conv4(v5)
        v7 = self.conv5(v6)
        v7_pool = self.pool(v7, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        v8 = self.conv6(v7_pool)
        v9 = self.conv7(v8)
        v10 = self.conv8(v9)
        return v10
    @staticmethod
    def pool(x, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
        result = F.avg_pool2d(F.pad(x, (padding, padding, padding, padding)), (kernel_size, kernel_size), stride=stride, padding=(0, 0), ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        return result
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
