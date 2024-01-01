
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0)
        self.c2 = torch.nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        self.c3 = torch.nn.Conv2d(12, 24, kernel_size=1, stride=1, padding=0)
        self.c4 = torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.c5 = torch.nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.c6 = torch.nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.c7 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.c8 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.c9 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.c10 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.c11 = torch.nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.c1(x1)
        v2 = self.c2(v1)
        v3 = self.c3(v2)
        v4 = self.c4(v3)
        v5 = self.c5(v4)
        v6 = self.c6(v5)
        v7 = v6+v6
        v8 = self.c7(v7)
        v9 = self.c8(v8)
        v10 = self.c9(v9)
        v11 = self.c10(v10)
        v12 = self.c11(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 255, 255)
