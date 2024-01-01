
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        xx1 = torch.mm(x1, x2)
        xx2 = torch.mm(x3, x4)
        xx3 = torch.mm(x1, x2)
        xx4 = torch.mm(x3, x4)
        xx5 = torch.mm(x1, x2)
        xx6 = torch.mm(x3, x4)
        xx7 = torch.mm(x1, x2)
        xx8 = torch.mm(x3, x4)
        xx9 = torch.mm(x1, x2)
        xx10 = torch.mm(x3, x4)
        xx11 = torch.mm(x1, x2)
        xx12 = torch.mm(x3, x4)
        xx13 = torch.mm(x1, x2)
        xx14 = torch.mm(x3, x4)
        xx15 = torch.mm(x1, x2)
        xx16 = torch.mm(x3, x4)
        xx17 = xx1 + xx2
        xx18 = xx17 + xx3
        xx19 = xx18 + xx4
        xx20 = xx19 + xx5
        xx21 = xx20 + xx6
        xx22 = xx21 + xx7
        xx23 = xx22 + xx8
        xx24 = xx23 + xx9
        xx25 = xx24 + xx10
        xx26 = xx25 + xx11
        xx27 = xx26 + xx12
        xx28 = xx27 + xx13
        xx29 = xx28 + xx14
        xx30 = xx29 + xx15
        xx31 = xx30 + xx16
        xx32 = xx31 + xx17
        return xx32
# Inputs to the model
x1 = torch.randn(1, 65)
x2 = torch.randn(65, 5)
x3 = torch.randn(1, 65)
x4 = torch.randn(65, 5)
