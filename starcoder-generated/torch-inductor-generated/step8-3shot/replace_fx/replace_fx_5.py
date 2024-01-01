
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = x1 + x2
        x4 = torch.nn.functional.dropout(x3, p=0.1, training=False)
        x5 = torch.rand_like(x4)
        a6 = x4 * x5
        a7 = torch.nn.functional.dropout(a6, p=0.2)
        a8 = torch.rand_like(x1)
        a9 = a6 - a8
        a10 = torch.rand_like(x2)
        a11 = a9 - a10
        a12 = torch.nn.functional.dropout(a11)
        a13 = torch.nn.functional.dropout(a11, p=0.6)
        a14 = torch.nn.functional.dropout(x2, p=0.6)
        a15 = torch.nn.functional.dropout(x3, p=0.6)
        a16 = torch.pow(a12, 2)
        a17 = a12 - a13
        a18 = a12 - a14
        a19 = a11 - a15
        a20 = a16 ** 2
        a21 = a17 ** 2
        a22 = a18 ** 2
        a23 = a19 ** 2
        a24 = a20 * a21
        a25 = (x3 - a22) / 2
        a26 = x2 ** 2
        a27 = a21 - a23
        a28 = a23 + a24
        a29 = x1 * a25
        a30 = x2 / a14
        a31 = torch.nn.functional.gelu(a30)
        a32 = a11 - a20
        a33 = torch.nn.functional.gelu(a31)
        a34 = x3 - a24
        a35 = torch.nn.functional.dropout(a31)
        a36 = torch.nn.functional.dropout(a31, p=0.7)
        a37 = nn.Sigmoid()(a27)
        a38 = nn.GELU()(x4)
        a39 = a37 + a38
        a40 = a39.mean()
        a41 = x5 * a34
        return a12
# Inputs to the model
x1 = torch.randn(10)
x2 = torch.randn(10)
