
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.add(v1, 3)
        v2 = torch.clamp(v1, 0, 6)
        v3 = torch.div(v2, 6)
        v4 = self.other_conv(v3)
        v4 = torch.add_(v4, 3)
        v5 = torch.clamp_(v4, 0, 6)
        v6 = torch.div_(v5, 6)
        return v6
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
