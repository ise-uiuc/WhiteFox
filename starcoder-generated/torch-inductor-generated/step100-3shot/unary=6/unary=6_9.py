
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 48, 1, stride=1, padding=1)
    def forward(self, x1):
        s1 = self.conv(x1)
        s2 = 3 + s1
        s3 = torch.clamp(s2, min=0, max=6)
        s4 = torch.nn.functional.softmin(s1)
        s5 = s3 * s4
        s6 = torch.nn.functional.softmax(s5, dim=-1)
        s7 = s6 / 6
        return s7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
