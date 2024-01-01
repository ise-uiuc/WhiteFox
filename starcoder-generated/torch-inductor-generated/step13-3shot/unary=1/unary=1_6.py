
class Model(torch.nn.Module):
    def __init__(self, i1, i2):
        super().__init__()
        self.linear = torch.nn.Linear(i1, i2)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 * 0.5
        v4 = v2 + (v2 * v2 * v2) * 0.044715
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v3 * v7
        return v8

# Initializing the model
__i_s__ = (2048, 2048)
m = Model(*__i_s__)

# Inputs to the model
x2 = torch.randn(1, *__i_s__)
