
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min, out=torch.jit.annotate(torch.Tensor, None))
        v3 = torch.clamp_max(v2, self.max, out=torch.jit.annotate(torch.Tensor, None))
        return v3
min = 1.5
max = 1.8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 128)
