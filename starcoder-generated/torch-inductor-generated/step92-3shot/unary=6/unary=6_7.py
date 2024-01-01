
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 27, 1, bias=False, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1.sum(dim=1, keepdim=True)   # or t2 = torch.sum(t1, dim=[1], keepdim=True)
        t3 = torch.relu6(t2 + 3)
        t4 = torch.clamp_max(2*t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(2, 3, 108, 64)
