
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        i1 = torch.mm(x1, x2)
        i1_c = torch.cat(
            [i1, i1, i1, i1, i1, i1, i1, i1, i1], 1)

        i2 = torch.mm(x1, x2)
        i2_c = torch.cat(
            [i2, i2, i2, i2, i2, i2, i2, i2, i2], 1)
        i3 = torch.mm(x1, x2)
        i3_c = torch.cat(
            [i3, i3, i3, i3, i3, i3, i3, i3, i3], 1)
        i4 = torch.mm(x1, x2)
        i4_c = torch.cat(
            [i4, i4, i4, i4, i4, i4, i4, i4, i4], 1)
        i = torch.cat(
            [i1_c, i2_c, i3_c, i4_c],
            1)
        return i
# Inputs to the model
x1 = torch.randn(6, 7)
x2 = torch.randn(7, 4)
