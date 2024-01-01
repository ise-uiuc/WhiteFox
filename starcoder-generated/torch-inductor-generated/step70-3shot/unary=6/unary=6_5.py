
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        w1 = self.conv(x1)
        w2 = w1 + 3
        w3 = torch.clamp(w2, 0, 6)
        w4 = w3 * w1
        w5 = w4 / 6
        return w5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
