
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.15470053837925458
        v3 = v1 * 0.5502276670089287
        v4 = torch.erf(v3)
        v5 = v4 + 0.34449133415077183
        v6 = v1 * 3.3784414133109406
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
