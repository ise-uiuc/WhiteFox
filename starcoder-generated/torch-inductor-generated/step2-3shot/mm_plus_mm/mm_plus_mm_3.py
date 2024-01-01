
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        x5 = torch.rand(1, 4)
        x6 = torch.rand(1, 2)
        v3 = v1 + v2
        v4 = torch.mm(x5, x6)
        v5 = v3 + v4
        r1 = torch.abs(v2 + v5)
        r2 = torch.max(v1 + v4 + r1, r1 + r1 * r1, r1 * r1 - v3)
        v6 = v1 / v6
        s1 = torch.nn.functional.sigmoid(0.2 * s2 - 3.4 * s1)
        s2 = torch.nn.functional.sigmoid(s2 - s1) * s1 * \
             torch.nn.functional.sigmoid(0.3 * s1 + 1.4 * s2)
        return v6
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(4, 4)
x4 = torch.randn(3, 3)
