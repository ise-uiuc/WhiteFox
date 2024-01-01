
class Model(torch.nn.Module):
    def __init__(self, min_clamp=0.0, max_clamp=0.9):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 2, stride=3, padding=10)
        self.min_clamp = torch.zeros(1) + min_clamp
        self.max_clamp = torch.zeros(1) + max_clamp
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_clamp)
        v3 = torch.clamp_max(v2, self.max_clamp)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 50, 50)
