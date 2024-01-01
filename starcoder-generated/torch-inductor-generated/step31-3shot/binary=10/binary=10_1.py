
class Model(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
        self.a = a
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.a
        return v2

# Initializing the model
a = torch.nn.Parameter(torch.randn(8), requires_grad=True)
m = Model(a)

# Inputs to the model
x1 = torch.randn(4)
