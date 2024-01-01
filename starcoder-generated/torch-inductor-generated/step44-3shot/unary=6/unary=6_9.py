
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.pool = torch.nn.AvgPool3d(3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.convolution(x1)
        t1 = self.pool(v1)
        t2 = 3 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 30, 104, 256)
