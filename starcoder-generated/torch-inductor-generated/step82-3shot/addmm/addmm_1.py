
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, x4)
        v2 = torch.mm(x3, v2)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3)
