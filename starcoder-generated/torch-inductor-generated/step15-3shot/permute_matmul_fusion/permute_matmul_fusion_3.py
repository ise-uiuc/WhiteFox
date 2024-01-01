
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t0 = x1.permute(0, 2, 1)
        t1 = x2.permute(0, 2, 1)
        t2 = torch.bmm(t0, t1)
        t3 = x2 * t2
        return t3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
