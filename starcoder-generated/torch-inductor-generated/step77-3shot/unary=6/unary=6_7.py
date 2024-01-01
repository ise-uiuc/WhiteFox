
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = 3 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        return self.conv(t4)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
