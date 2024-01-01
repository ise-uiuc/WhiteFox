
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, A):
        v1 = torch.mm(x1, A)
        return v1 + x2
# Inputs to the model
x1 = torch.randn(3, 8, requires_grad=True)
x2 = torch.randn(3, 8, requires_grad=True)
A = torch.randn(8, 8, requires_grad=True)
