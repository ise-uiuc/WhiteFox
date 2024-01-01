
class Model(torch.nn.Module):
    def __init__(self, min_clamp=0.5, max_clamp=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 2, stride=1, padding=0)
        self.min_clamp = torch.zeros(1) + min_clamp
        self.max_clamp = torch.zeros(1) + max_clamp
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, min=self.min_clamp, max=self.max_clamp)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 28, 28)
