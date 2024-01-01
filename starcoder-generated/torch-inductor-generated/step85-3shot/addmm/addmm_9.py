
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1):
        v1 = torch.mm(x1, x2)
        return torch.add(v1, inp1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp1 = torch.randn(3, 3, requires_grad=True)
