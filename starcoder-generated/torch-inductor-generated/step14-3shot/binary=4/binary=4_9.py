
class Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
linear = torch.nn.Linear(2, 3)
m = Model(linear)
 
# Inputs to the model
x1 = torch.randn(16, 2)
x2 = torch.randn(16, 3)
