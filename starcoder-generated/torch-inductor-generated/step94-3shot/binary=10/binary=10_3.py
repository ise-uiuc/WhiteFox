
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=True)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(torch.randn(16, 16))

# Inputs to the model
x = torch.randn(16, 16)
