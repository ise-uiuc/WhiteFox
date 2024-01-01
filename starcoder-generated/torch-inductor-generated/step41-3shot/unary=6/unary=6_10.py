
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 3, 7, stride=2, padding=(3, 3))
    def forward(self, x1):
        t1 = self.conv2(x1)
        t2 = torch.clamp(3 + t1, 0, 6)
        t3 = torch.clamp(t1, 0, 6)
        t4 = t2 / t3
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
