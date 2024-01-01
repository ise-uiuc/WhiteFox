
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        res1 = x1 + x2
        res2 = torch.mm(res1, res1)
        res2 = res1 * res2
        res2 = res2 + x1
        v = torch.mm(res1, res1)
        v = v + x1 * res2 + x1
        return (res2, v)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
