
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x.numel()
        w = torch.ones(a, dtype=torch.float32)
        r = torch.randperm(a)
        w[r[a:]] = -1e9
        x = x*torch.rsqrt(torch.matmul(x, torch.matmul(w, w)))
        del w, r, a
        y = torch.cat([x,x], dim=0)
        return y.view(2, -1)
# Inputs to the model
x = torch.randn(2, 3)
