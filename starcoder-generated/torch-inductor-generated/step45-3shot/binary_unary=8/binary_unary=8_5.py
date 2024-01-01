
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 32, 1, 1, 0, 1, 1, bias=False)
        self.c2 = torch.nn.Conv2d(32, 32, 1, 1, 0, 1, 1, bias=False)
        self.c3 = torch.nn.Conv2d(32, 32, 3, 1, 1, 1, 1, bias=False, dilation=2)
        self.c4 = torch.nn.Conv2d(32, 32, 1, 1, 0, 1, 1, bias=False)
        self.c5 = torch.nn.Conv2d(32, 32, 3, 1, 2, 1, 1, bias=False)
    def forward(self, x):
        v10 = self.c1(x)
        v1 = self.c2(v10)
        v2 = self.c1(x)
        v3 = self.c2(v2)
        v4 = self.c3(v3)
        v5 = v1 + v2 + v3 + v4
        v6 = self.c1(x)
        v7 = self.c2(v6)
        v8 = self.c1(x)
        v9 = self.c2(v8)
        v5 = torch.nn.functional.relu(v5)
        v7 = torch.nn.functional.relu(v7)
        v9 = torch.nn.functional.relu(v9)
        v4 = self.c4(v5)
        v10 = v4 + v9
        v10 = self.c4(v10)
        v3 = self.c5(v10)
        v3 = torch.nn.functional.relu(v3)
        return v3
# Inputs to the model:
x = torch.randn(1, 1, 224, 224)
