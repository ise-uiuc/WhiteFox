
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        l1 = torch.cat([v6, v1, v3], 1)
        l2 = torch.transpose(l1, 0, 1)
        n1 = torch.flatten(l2, start_dim=1)
        return n1
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
