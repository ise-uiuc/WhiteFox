
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=True)
 
    def forward(self, x1):
        x2 = linear(x1)
        o1 = x2.clamp(min=0, max=6)
        return o1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 16)
