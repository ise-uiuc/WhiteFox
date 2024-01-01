
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1 + x2
        t2 = t1 * 2
        t3 = t2 + 3
        t4 = torch.rand_like(t3)
        t5 = torch.rand_like(x1)
        x, _ = torch.topk(t5, 5)
        return t3 + t4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = 1
