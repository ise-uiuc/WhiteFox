
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(x1, x2) + inp1
        v2 = v1 + inp2
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 6, 12)
x2 = torch.randn(1, 1, 12, 6)
inp1 = torch.randn(6, 6)
inp2 = torch.randn(6, 6)
