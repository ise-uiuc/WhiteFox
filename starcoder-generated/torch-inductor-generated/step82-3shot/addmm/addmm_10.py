
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        m = torch.mm(x1, x2.transpose(1, 0))
        return torch.mm(m, x3)
# Inputs to the model
x1 = torch.randn(3, 4, requires_grad=True)
x2 = torch.randn(4, 5, requires_grad=True)
x3 = torch.randn(5, 2, requires_grad=True)
