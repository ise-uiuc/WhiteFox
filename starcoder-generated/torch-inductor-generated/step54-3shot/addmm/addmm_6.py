
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x2, x3)
        v1 = torch.add(v1, x1)
        v1[0][0] = x1[0][0] + x2[0][0] + x3[0][0]
        v1[1][1] = x1[0][0] + x2[1][1] + x3[1][1]
        v1[2][2] = x1[0][0] + x2[2][2] + x3[2][2]
        return v1 + inp
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
inp = torch.randn(3, 3)
