
class Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 131.7624408748667
        v3 = torch.gelu(v2)
        return v3

# Initializing the model
linear = torch.nn.Linear(3, 512, 1)
m = Model(linear)

# Inputs to the model
x1 = torch.randn(1, 3)
