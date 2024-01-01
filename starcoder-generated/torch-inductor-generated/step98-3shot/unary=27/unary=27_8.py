
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.l1 = torch.nn.Conv2d(18, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(2, eps=1e-05, momentum=1.0, affine=True)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = self.bn1(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.06
max = 0.11
# Inputs to the model
x1 = torch.randn(1, 18, 10, 10)
