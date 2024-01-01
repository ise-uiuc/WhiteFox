
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 13, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(13, 7, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(7, 7, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(7, 8, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(8, 5, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(5, 6, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(6, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v3_pool = self.pool(v3, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        v4 = self.conv3(v3_pool)
        v5 = self.conv4(v4)
        v5_pool = self.pool(v5, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        v6 = self.conv5(v5_pool)
        v7 = self.conv6(v6)
        v8 = self.conv7(v7)
        v9 = self.conv8(v8)
        v10 = torch.relu(v9)
        return v10
    def pool(self, x, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True):
        result = F.avg_pool2d(F.pad(x, (padding, padding, padding, padding)), (kernel_size, kernel_size), stride=stride, padding=(0, 0), ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        return result
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
