
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.w = torch.rand(8, 3, 1, 1)
        self.b = torch.rand(8)
 
    def forward(self, x1):
        v1 = F.conv2d(x1, self.w, self.b, 1, 1, 1, 1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5