
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        a1 = torch.matmul(x1, x2.transpose(-2, -1))
        b1 = a1 / x3
        c1 = torch.nn.functional.dropout(b1, x4, training=True)
        d1 = torch.matmul(c1, x5)
        b2 = d1.add(x6)
        c2 = torch.nn.functional.gelu(b2)
        f1 = torch.matmul(c2, x7)
        h1 = torch.nn.functional.dropout(f1, x8, training=True)
        g1 = torch.sigmoid(h1)
        return g1

