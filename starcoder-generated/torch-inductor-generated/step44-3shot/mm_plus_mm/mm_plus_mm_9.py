
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m = torch.nn.Linear(2, 7)

    def forward(input, x1):
        v1 = torch.mm(x1, x4)
        v2 = torch.mm(x3, x2)
        v3 = v1 + v2
        return v3
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
