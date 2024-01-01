
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp 
        v2 = v2 * self.t
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
