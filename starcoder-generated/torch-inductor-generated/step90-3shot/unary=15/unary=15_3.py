
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dWithReLU(2, 32, 2, 2)
        self.conv2 = Conv2dWithReLU(32, 64, 1, 1, 1, 1)
        self.conv3 = Conv2dWithReLU(64, 64, 1, 1, 1, 1)
        self.conv4 = Conv2dWithReLU(64, 128, 1, 1, 1, 1)
        self.conv5 = Conv2dWithReLU(128, 128, 1, 1, 1, 1)
        self.conv6 = Conv2dWithReLU(128, 32, 1, 1, 1, 1)
        self.conv7 = Conv2dWithReLU(32, 8, 1, 1, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = torch.clamp(v6, -1, 1)
        v27 = v7.matmul(torch.nn.init.xavier_normal_(torch.empty(8, 8)))
        v8 = self.conv7(v27)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 12, 12)
