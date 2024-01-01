
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        t2 = x2.permute(0, 2, 1)
        y1 = torch.bmm(t1, t2)
        t1 = y1.permute(0, 2, 1)
        r1 = torch.matmul(t1, x2)
        return y1, r1
# Inputs to the model
x1 = torch.randn(1, 3, 2)
x2 = torch.randn(1, 3, 2)
