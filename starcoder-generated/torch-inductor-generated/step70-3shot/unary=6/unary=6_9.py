
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2, padding=3)
        self.conv1 = torch.nn.Conv2d(3, 16, 2, stride=1, padding=1)
    def forward(self, x1):
        s1 = self.conv(x1)
        s2 = s1 + 3  # q1
        s3 = torch.clamp(s2, 0, 6)  # r1
        s4 = s1 * s3  # r2
        s5 = s4 / 6  # r3
        s6 = self.conv1(s5)
        s7 = s6 + 7  # s1
        s8 = torch.clamp(s7, 0, 16)  # s2
        s9 = s6 * s8  # s3
        s10 = s9 / 16  # s4
        s11 = torch.exp(s10)  # t1
        s12 = s11 / 12  # t2
        return s12, s1, r2, r3, s11, s2, s4, t1, t2, s3, s8, s12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
