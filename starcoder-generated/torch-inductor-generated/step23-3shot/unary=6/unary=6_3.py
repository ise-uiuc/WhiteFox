
class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        t1 = F.conv2d(x1, weight=np.zeros((3, 3, 1, 1)), stride=1, padding=1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6

x1 = torch.randn(1, 3, 64, 64)
