
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        return torch.mm(x1, inp)
# Inputs to the model
x1 = torch.randn(3, 3)
inp = torch.randn(3, 3)
