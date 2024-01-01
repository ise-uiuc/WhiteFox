
class Model(torch.nn.Module):
    def forward(self, x):
        e = torch.empty_like(x)
        e[..., 5] = 99.999
        _t = e[..., 5:]
        t1 = torch.mm(x, x)
        t2 = torch.mm(x, x)
        t3 = t1 + t2
        t3 = t3 + e
        t3 = t3 + e
        t3 = t3 + e
        t3 = t3 + e
        t3 = t3 + e
        return t3, t1, t2
# Inputs to the model
x = torch.rand(3, 3)
