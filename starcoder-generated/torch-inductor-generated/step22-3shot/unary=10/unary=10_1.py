
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 32, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __v1__ = self.linear.bias
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()
m.linear.bias = __v1__

# Inputs to the model
x1 = torch.randn(1, 224)
