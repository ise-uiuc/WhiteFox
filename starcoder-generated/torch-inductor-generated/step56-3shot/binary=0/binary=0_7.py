
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, other=None, padding1=None, bias=True):
        v1 = x1 + other
        v2 = torch.add(v1, 1)
        if bias:
            v2 += padding1
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
