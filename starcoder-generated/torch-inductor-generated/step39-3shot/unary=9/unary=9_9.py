
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.alpha = torch.randn(8, 8, 1, 1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = nn.functional.pad(t1, (0,0,0,0,0,0,0,0,1,0,0,0))
        t3 = torch.mul(t2, self.alpha)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
