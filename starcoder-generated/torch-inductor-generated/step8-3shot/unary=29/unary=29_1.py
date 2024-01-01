
class Model(torch.nn.Module):
    def __init__(self, min_value=-5.171, max_value=-2.669):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 16, 1, bias=True, stride=1, padding=0)
        self.gelu = torch.nn.GELU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.convt(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.gelu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
