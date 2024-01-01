
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, inp)
        v2 = v1 * torch.sigmoid(x1) 
        return v2
# Inputs to the model
x1 = torch.randn(3, 3) * 2
x2 = torch.randn(3, 3) * 1.2 ** 2
inp = torch.randn(3, 3) * 1.2 ** 3
