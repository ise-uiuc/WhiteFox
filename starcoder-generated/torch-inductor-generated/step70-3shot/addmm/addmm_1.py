
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3, False)
        self.linear2 = torch.nn.Linear(3, 3, False)
    def forward(self, x1, x2, inp):
        v1 = x1 + x2
        v2 = self.linear2(v1)
        v3 = v2.addmm(x2, inp, beta=1, alpha=1)
        return self.linear1(v3)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=False)
x2 = torch.randn(3, 3, requires_grad=False)
inp = torch.randn(3, 3, requires_grad=True)
