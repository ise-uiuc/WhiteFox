
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(x1, inp1)
        v2 = torch.mm(v1, x2) + inp1
        v3 = torch.mm(x2, inp2)
        v4 = v3 - inp1
        v4 = v4 + v2
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
