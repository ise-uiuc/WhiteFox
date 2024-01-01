
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        return torch.mm(x1, x2) + inp + x1
# Inputs to the model
input1 = torch.randn(3, 4)
input2 = torch.randn(4, 5)
inp = torch.randn(5, 2)
