
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 6
        t3 = F.hardtanh(t2, min_val=0., max_val=6., inplace=False)
        t4 = t3 / 6.
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
