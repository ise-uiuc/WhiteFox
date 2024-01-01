
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp_min(t1, 0)
        t4 = torch.nn.functional.relu(t1)
        t5 = torch.clamp_max(t1, 6)
        t6 = t2 + t5
        t7 = t4 * t6
        t8 = t7 / 6
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
