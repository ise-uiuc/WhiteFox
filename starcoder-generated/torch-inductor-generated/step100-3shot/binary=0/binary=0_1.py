
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=1, padding=4)
    def forward(self, x1, padding1=False, padding2=torch.randint(-2, 3, [3, 7, 7]), padding3=2):
        v1 = self.conv(x1)
        v2 = v1 + 2
        t3 = torch.randint(0, 3, [3, 7, 7])
        v3 = v2 * t3
        t4 = torch.randint(0, 3, [3, 7, 7])
        v4 = v3 + t4
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 124, 124)
