
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=3, padding=3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = F.hardtanh(t2, min_val=0., max_val=6.)
        t4 = t1 * t3
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 200, 200)
