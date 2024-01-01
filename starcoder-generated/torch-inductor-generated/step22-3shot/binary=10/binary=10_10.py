
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 100, bias=False)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(120, 20)
y = torch.randn(100)
