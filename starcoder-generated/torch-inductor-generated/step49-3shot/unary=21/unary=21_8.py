
class ModelTanh(torch.nn.Module):
    def __init__(self, m, s):
        super().__init__()
        self.conv = torch.nn.Conv2d(m, s, 1, stride=1, padding=1)
    def forward(self, x2):
        y2 = self.conv(x2)
        t2 = torch.tanh(y2)
        return t2
x2 = torch.randn(1, 3, 64, 64)
