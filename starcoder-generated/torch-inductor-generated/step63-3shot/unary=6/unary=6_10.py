
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Note: We changed stride to 2 here to trigger some failures
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2, padding=2)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t2, 6)
        t5 = t3 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
