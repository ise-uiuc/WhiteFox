
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3
        t3 = torch.add(t1, t2)
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 7)
        t6 = torch.div(t5, 7)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
