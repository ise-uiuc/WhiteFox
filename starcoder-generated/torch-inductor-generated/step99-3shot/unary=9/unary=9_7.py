
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1) -> torch.Tensor:
        v1 = self._conv(x1)
        t0 = 3
        v2 = v1 + t0
        t1 = v2.clamp_min(0)
        t2 = t1.clamp_max(6)
        t3 = t2 / 6
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
