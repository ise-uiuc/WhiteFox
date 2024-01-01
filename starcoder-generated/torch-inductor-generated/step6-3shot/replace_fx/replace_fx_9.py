
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = torch.nn.functional.dropout(x1, training=True, p=0.4)
        b = torch.nn.functional.dropout(a, training=True, p=0.1)
        c = torch.nn.functional.dropout(b, training=True, p=0.0)
        d = torch.nn.functional.dropout(c, training=True, p=0.0)
        e = torch.nn.functional.dropout(d, training=True, p=0.2)
        f = torch.nn.functional.dropout(e, training=True, p=0.1)
        g = torch.rand_like(d)
        h = torch.random_like(d)
        i = g + d
        j = torch.nn.functional.dropout(f, training=True, p=0.0, inplace=False)
        k = h + d
        l = torch.nn.functional.dropout(k, training=True, p=0.4, inplace=True)
        return torch.nn.functional.dropout(l, training=True, p=0.1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
