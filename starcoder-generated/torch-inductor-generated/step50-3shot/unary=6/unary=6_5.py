
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.maxpool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.bn = torch.nn.BatchNorm1d(input_size)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.maxpool(t1)
        t3 = t2.reshape(3*192)
        t4 = 3 + t3
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = torch.mm(t4, t6) / 6
        t8 = self.bn(t7)
        return t8.unsqueeze(3).unsqueeze(4)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
