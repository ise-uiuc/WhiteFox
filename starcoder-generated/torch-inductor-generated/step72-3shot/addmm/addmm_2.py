
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp, x2):
        if inp.shape[0] == 2:
            v1 = torch.mm(x1, inp) + x2.flatten()
        elif inp.shape[0] == 5:
            v1 = torch.mm(x1, inp) + x2 + x2.flatten()
        v2 = torch.mm(x1, inp) + x2.flatten()
        v3 = torch.mm(x1, inp) + x2.flatten()
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(5, 3)
