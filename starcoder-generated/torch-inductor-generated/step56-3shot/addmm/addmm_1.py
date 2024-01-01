
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        return x1 + F.linear(x1, x2, inp)
# Inputs to the model
x1 = torch.randn(3)
x2 = torch.randn(3, 3)
inp = torch.randn(3)
