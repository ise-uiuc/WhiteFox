
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t0 = torch.bmm(x1, x2)
        t1 = x1.permute(0, 2, 1)
        t2 = t0.permute(0, 2, 1)
        t3 = t0.permute(0, 2, 1)
        t4 = t2 * t3
        return t4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
