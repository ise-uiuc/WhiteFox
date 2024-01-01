
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        d = OrderedDict([('a', torch.rand_like(x1)), ('b', torch.rand_like(x1))])
        return x1
# Inputs to the model
x1 = torch.randn(2, 3)
