
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2=None):
        if x2 is None:
            x3 = x1
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(v1, x1)
        v3 = v2 + x1
        v4 = v3 + x1
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
