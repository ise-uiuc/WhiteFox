
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 3, 3, bias=False, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(3)
        self.bn1.momentum = 0.01
        self.conv2 = torch.nn.Conv3d(3, 3, 3, bias=False, padding=2)
        self.conv2.groups = 3
        self.avg_pool = torch.nn.AvgPool3d((1,71,35))
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        y = self.avg_pool(x)
        return y
# Inputs to the model
x = torch.randn(1, 3, 32, 100, 100)
