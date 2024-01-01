
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        x2 = torch.mm(x1, x1)
        v1 = torch.mm(x1, x2) + inp
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
