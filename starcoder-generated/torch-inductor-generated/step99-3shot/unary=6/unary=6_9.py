
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = self.AdaptiveAvgPool2d(1, 1)
    def forward(self, x1):
        t1 = self.pool(x1) + 0.1
        t2 = 0.2 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6

class SpatialBatchNorm(torch.nn.Base):
    pass # to be completed

class BNN(torch.nn.Base):
    pass # to be completed

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = BNN(6, 8)
    def forward(self, x1):
        t1 = self.bn(x1)
        return t1

# Inputs to the model
x1 = torch.randn(1, 6, 4, 4)
