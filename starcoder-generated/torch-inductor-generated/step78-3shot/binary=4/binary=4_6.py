
class Model(torch.nn.Module):
    def __init__(self, v, w):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.v = v
        self.w = w
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.v
        return v2

# Initializing the model
v = torch.randn(1, 1)
w = torch.randn(1, 1)
m = Model(v, w)

# Inputs to the model
x1 = torch.randn(1, 16)
