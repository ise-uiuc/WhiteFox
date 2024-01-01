
class Model(torch.nn.Module):
    def __init__(self, i1, i2, i3, i4):
        super(Model, self).__init__()
        self.i1 = self.parameter.add_parameter(i1)
        self.i2 = i2
        self.i3 = parameter.add_parameter(i3)
        self.i4 = parameter.add_parameter(i4)
    def forward(self, x1, x2, x3, x4):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x2, x3)
        h3 = torch.mm(x3, x4)
        h4 = torch.mm(x4, self.i1)
        return h1 + h2 + h4
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
