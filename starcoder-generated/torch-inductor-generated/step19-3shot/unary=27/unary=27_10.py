
class Model(torch.nn.Module):
    def __init__(self, min=-6., max=-6.):
        super().__init__()
        self.t = torch.nn.Conv2d(4, 32, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        t1 = self.t(x1)
        t2 = torch.clamp_min(t1, self.min)
        t3 = torch.clamp_max(t2, self.max)
        return t3
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
