
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = torch.mm(x1, x2)
        b = torch.mm(x1, x2)
        c = torch.mm(x1, x2)
        d = torch.cat([a, b], 1)
        e = torch.cat([a, b, c], 1)
        f = torch.cat([d, e], 0)
        return f
