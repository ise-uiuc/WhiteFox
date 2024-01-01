
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(x1, inp2)
        v2 = v1 + inp1
        return v2
# Inputs to the model
x1 = torch.randn(0, 1)
x2 = torch.randn(1, 0)
inp1 = torch.randn(0, 0)
inp2 = torch.randn(1, 0)
