
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Parameter(torch.tensor(2.0))
    def forward(self, x1, x2, inp):
        v1 = torch.nn.ReLU()(torch.mm(x1, self.t1))
        v2 = torch.nn.Sigmoid()(torch.mm(x2, inp))
        return torch.mm(v1, v2)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
