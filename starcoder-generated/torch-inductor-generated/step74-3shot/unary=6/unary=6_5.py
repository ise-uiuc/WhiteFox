
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, stride=1, padding=2, bias=False)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.add(t1, 5)
        t3 = torch.clamp(t2, 2, 6)
        t4 = torch.mean(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
