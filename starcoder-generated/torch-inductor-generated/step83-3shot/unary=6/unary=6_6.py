
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return (v1 + 3).clamp_(0, 6).mul(0.125).add(0.103125).div(0.20625)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
