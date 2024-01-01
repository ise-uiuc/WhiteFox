
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(134, 96, 5, stride=1, padding=2)
        self.t1 = torch.randn(1, 134, 68, 30)
    def forward(self, x1):
        y1 = self.t1 + 0.3
        y2 = torch.abs(y1)
        r1 = torch.clamp(y2, min=0)
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 134, 68, 30)
