
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.where(t2 > 0, t2, torch.tensor([5.0], dtype=t2.dtype, device=t2.device))
        t4 = torch.where(t3 < 6, t3, torch.tensor([5], dtype=t3.dtype, device = t3.device))
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
