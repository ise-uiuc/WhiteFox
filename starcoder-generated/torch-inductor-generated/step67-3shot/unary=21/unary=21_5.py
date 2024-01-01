
class Model(Module):
    def __init__(self, weight):
        super(Model, self).__init__()
        self.param_0 = Parameter(weight)
    def forward(self, x):
        x = x.mul(self.param_0)
        return x
# Inputs to the model
x = torch.randn(1, 3, 177, 14)
weight = torch.randn(3)
