
class Model(torch.nn.Module):
    def __init__(self, inp, oup, act=None):
        super().__init__()
        self.linear = torch.nn.Linear(inp, oup)
        self.act = act
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1
        if self.act:
            v2 = self.act(v1)
        return v2

# Initializing the model
m = Model(64, 64)

# Inputs to the model
x1 = torch.randn(1, 64)
