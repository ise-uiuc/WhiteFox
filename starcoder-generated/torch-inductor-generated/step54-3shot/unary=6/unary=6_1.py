
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t3 = F.hardtanh(t1, min_val=0.0, max_val=6.0)
        t4 = t3 / 6
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
