
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        t1 = self.bn(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
