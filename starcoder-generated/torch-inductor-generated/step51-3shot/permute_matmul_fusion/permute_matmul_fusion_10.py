
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = x1.permute(0, 2, 1)
        b = x1.permute(1, 0, 2)
        c = x1.permute(1, 2, 0)
        d = x1.permute(2, 0, 1)
        e = x1.permute(2, 1, 0)
        v1 = c.permute(0, 2, 1)
        v2 = torch.bmm(d, e.permute(0, 2, 1))
        v3 = torch.matmul(a, v1)
        v4 = v3.permute(0, 2, 1).contiguous()
        return v4.detach()
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
