
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(7, 7, 3),
            torch.nn.BatchNorm2d(7),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(7, 7, 3),
            torch.nn.BatchNorm2d(7),
            torch.nn.BatchNorm2d(7),
        )
    def forward(self, x2):
        s2 = self.block1(x2)
        s2 = self.block2(s2)
        y2 = s2 + s2
# Inputs to the model
x2 = torch.randn(1, 7, 6, 6)
