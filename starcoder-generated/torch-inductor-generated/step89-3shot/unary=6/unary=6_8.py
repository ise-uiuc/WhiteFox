
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 1, stride=1, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 1, stride=1, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(3, 3, 1, stride=1, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 1, stride=1, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 1, stride=1, padding=1),
            torch.nn.ReLU(),
        )
    def forward(self, x1):
        t1 = self.block1(x1)
        t2 = t1 * t1
        t3 = self.block2(x1)
        t4 = t3 + t3
        t5 = t2 + t4
        t6 = self.block3(x1)
        t7 = t6 * t1
        t8 = self.block4(x1)
        t9 = t8 + t8
        t10 = t7 + t9
        return t5 + t10
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
