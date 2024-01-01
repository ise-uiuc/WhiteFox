
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + torch.full((3,), 3, dtype=v1.dtype, device=v1.device, requires_grad=False)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.full((3,), 6, dtype=v1.dtype, device=v1.device, requires_grad=False)
        v5 = torch.max(v3, v4)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
