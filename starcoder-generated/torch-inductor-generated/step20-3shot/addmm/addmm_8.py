
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(x2, x1)
        v2 = inp1
        v2 = v1 + v2
        v2 = v2 + inp2
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(5, 3)
inp1 = torch.randn(5, 3, 3)
inp2 = torch.randn(3, 3, 5)
