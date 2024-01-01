
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        p1 = self.conv(x1)
        p2 = p1 + 3
        p3 = torch.nn.functional.hardtanh(p2, min_val=0)
        p4 = torch.nn.functional.hardtanh(p2, max_val=6)
        p5 = p1 * p4
        p6 = p5 / 6
        return p6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
