
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.transpose(1, 2)
        v3 = v2.div(2)
        v4 = (v3 - 2.1).abs()
        v5 = torch.clamp(v4, min=0.1)
        v6 = torch.clamp(v5, max=0.9)
        v7 = v6.transpose(2, 1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
