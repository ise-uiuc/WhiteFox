
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        t = a + b
        v = torch.mm(t, t)
        return t, v
# Inputs to the model
a = torch.randn(3, 3, requires_grad=True)
b = torch.randn(3, 3, requires_grad=True)
