
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)  # 3x3
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)  # 1x1
        # self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)  # 1x1
    def forward(self, x1):
        t1 = self.conv(x1)  # 3x3
        t2 = self.conv2(t1)  # 1x1
        # t3 = self.conv3(t2)  # 1x1
        return t2
# Inputs to the model
x1 = torch.randn(1, 16, 10, 10)
