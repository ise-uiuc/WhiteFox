
class Model(torch.nn.Module):
    def __init__(self, min_r, max_r, min_g, max_g, min_b, max_b):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.min_r = min_r
        self.max_r = max_r
        self.min_g = min_g
        self.max_g = max_g
        self.min_b = min_b
        self.max_b = max_b
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, min=self.min_r, max=self.max_r)
        v3 = torch.clamp(v2, min=self.min_g, max=self.max_g)
        v4 = torch.clamp(v3, min=self.min_b, max=self.max_b)
        return v4
min_red= 0.3
max_red= 0.7
min_green= 0.8
max_green= 0.8
min_blue= 0.6
max_blue= 0.6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
