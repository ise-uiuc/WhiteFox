
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x, inp):
        x1 = self.bn(x)
        v1 = self.linear(x1)
        v2 = v1.detach() + x1
        v3 = torch.mm(v2, v2)
        return (v1, v3 + inp, v1.detach() + x)
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
