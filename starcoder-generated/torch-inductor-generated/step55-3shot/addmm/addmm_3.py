
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp, x2, x1):
        v1 = torch.mm(inp, x2)
        v1 = v1 + x1
        vv1 = torch.mul(x1, x2)
        v1[0][0] = vv1[0][0]
        v1[1][1] = vv1[1][1]
        v1[2][2] = vv1[2][2]
        return v1
# Inputs to the model
inp = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x1 = torch.randn(3, 3)
