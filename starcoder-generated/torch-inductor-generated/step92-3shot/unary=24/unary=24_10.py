
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = torch.nn.Conv2d(3, 11, 1, stride=(2, 1), padding=0, groups=1)
        self.c1 = torch.nn.Conv2d(11, 41, (2, 1), stride=(1, 1), padding=(2, 0), groups=1)
        self.c2 = torch.nn.Conv2d(41, 53, (1, 1), stride=(1, 1), padding=0, groups=1)
        self.c3 = torch.nn.Conv2d(53, 50, (1, 2), stride=(1, 1), padding=0, groups=1)
        self.c4 = torch.nn.Conv2d(50, 57, (2, 2), stride=(2, 1), padding=(1, 0), groups=1)
    def forward(self, x):
        negative_slope = 0.19527944
        v1 = self.c0(x)
        v2 = self.c1(v1)
        v3 = self.c2(v2)
        v4 = self.c3(v3)
        v5 = self.c4(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 61, 319)
