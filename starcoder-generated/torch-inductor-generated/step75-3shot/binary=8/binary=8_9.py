
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = F.leaky_relu(v2, negative_slope=0.0587071712737512, inplace=True)
        v4 = self.bn1(v3)
        v5 = F.adaptive_avg_pool2d(v4, (1, 1))
        v6 = v5.view(1, -1)
        v7 = v6.mul(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)
x2 = torch.randn(2, 3, 112, 112)
