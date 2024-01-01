
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, weight, bias):
        t1 = torch.mm(x, weight.t())
        return t1
# Inputs to the model
x = torch.randn(1, 1)
weight = torch.randn(4, 1)
bias = torch.randn(4)
