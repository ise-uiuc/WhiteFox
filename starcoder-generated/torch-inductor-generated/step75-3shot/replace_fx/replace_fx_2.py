
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.rand_like(x1)
        t2 = torch.rand_like(x2)
        t3 = torch.rand_like(t1)
        t4 = torch.rand_like(t2)
        x3 = torch.rand_like(t1) + torch.rand_like(t2) * 2.0
        x3 += t1 * 3.0 + t2 * 4.0
        x3[-1] += t1[0, 0, 0] * 5.0
        x4 = t3 + t4 * 2.0
        return (x3, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 2, 2)
