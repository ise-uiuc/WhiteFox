


class Model(torch.nn.Module):
    def __init__(self, batch_norm):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 32, 5, stride=3, padding=0)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        f1 = self.conv_t(x1)
        if hasattr(self, 'bn'):
            f1 = self.bn(f1)
        f2 = f1 > 0
        f3 = f1 * 0.86
        f4 = torch.where(f2, f1, f3)
        return f4


class AnotherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = Model(True)
        self.m2 = Model(False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1):
        f1 = self.m1(x1)
        f2 = self.m2(x1)
        f3 = torch.stack([f1, f2])
        f4 = torch.min(f3)
        f5 = f3 > f4
        f6 = f3 * f5
        f7 = torch.prod(f6, dim=0)
        f8 = self.sigmoid(f7)
        return f8

# Inputs to the model
x1 = torch.randn(1, 15, 117, 173)
