
class MatrixAddition(torch.nn.Module):
    def __init__(self, x1, x2, x3, x4):
        super().__init__()
        self.m1 = torch.nn.Linear(x1)
        self.m2 = torch.nn.Linear(x2)
        self.m3 = torch.nn.Linear(x3)
        self.m4 = torch.nn.Linear(x4)

    def forward(self, x1, x2, x3, x4):
        y1 = self.m1(x1)
        y2 = self.m3(x1)
        y3 = self.m2(x2)
        y4 = self.m4(x4)

        y5 = self.m1(x2)
        y6 = self.m3(x2)

        y7 = self.m2(x3)
        y8 = self.m4(x3)

        return torch.mm(y1, y2) + torch.mm(y3, y4) + torch.mm(y5, y6) + torch.mm(y7, y8)
# Inputs to the model
x1 = torch.randn(4, 5)
x2 = torch.randn(3, 5)
x3 = torch.randn(5, 5)
x4 = torch.randn(3, 4)
