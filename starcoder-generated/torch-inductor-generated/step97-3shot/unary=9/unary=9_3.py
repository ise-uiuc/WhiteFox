
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        t1 = self.conv(x1)
        t2 = x2 + 3
        t3 = torch.clamp(t1 + t2, min=0, max=6)
        t4 = torch.div(t3, 6)
        return t4
# Input 1 to the model
x1 = torch.randn(1, 3, 64, 64)
# Input 2 to the model
x2 = torch.randn(1, 3, 64, 64)
