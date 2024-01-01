
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 3)
        conv2d_0_weight = torch.ones(8, 3, 3) * 5
        self.conv2d.weight = torch.nn.Parameter(conv2d_0_weight)
        self.conv2d_1 = torch.nn.Conv2d(8, 4, 2)
        conv2d_1_weight = torch.ones(4, 1, 1) * 5
        self.conv2d_1.weight = torch.nn.Parameter(conv2d_1_weight)
        self.batchnorm2d = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv2d_1(v1)
        v3 = v2.mean(-1)
        v4 = v3.permute(0, 2, 3, 1)
        v5 = torch.reshape(v4, (1, -1))
        v6 = self.batchnorm2d(v5)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 5, 6)
