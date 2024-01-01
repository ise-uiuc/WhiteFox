
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3, affine=False)
        self.linear = torch.nn.Linear(3, 3, False)
    def forward(self, x, inp):
        x = self.bn(x)
        v1 = self.linear(x)
        v2 = torch.mm(v1, v2)
        inp1 = torch.mm(v2, v2) + v1
        inp2 = v1 + v1
        return True
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
