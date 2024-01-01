
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = inp + x1
        v2 = torch.mm(v1, x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 28)
x2 = torch.randn(1, 28)
inp = torch.randn(1, 28)
