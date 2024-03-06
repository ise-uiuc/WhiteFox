
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = 3 + x1
        v1 = self.conv(x2)
        v2 = v1 + x1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6.permute(2, 3, 1, 0).unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 6, 128, 128)