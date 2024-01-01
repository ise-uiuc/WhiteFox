
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, i0, i1, y, x2):
        o1 = torch.nn.functional.gelu(y)
        v2 = i0 * i1
        v2 = torch.mm(o1, v2)
        o1 = torch.nn.functional.gelu(v2)
        torch.nn.functional.gelu(o1)
        o1 = torch.nn.functional.gelu(v2)
        o1 = torch.matmul(v2, i0)
        o1 = torch.matmul(o1, i1)
        o1 = o1 + o1
        o = o1 + v2
        return o
# Inputs to the model
i0 = torch.randn(6, 6, requires_grad=True)
i1 = torch.randn(6, 6, requires_grad=True)
y = torch.randn(6, 6, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
