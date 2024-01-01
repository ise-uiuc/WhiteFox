
class Model(torch.nn.Module):
    def __init__(self, in_out):
        super().__init__()
        self.linear = torch.nn.Linear(in_out, in_out)
 
    def forward(self, x, other=1):
        v1 = self.linear(x)
        v2 = v1 + other
        v2 = v2.relu()
        return v2

# Input tensors
x = torch.randn(1, 8, 3, 3)
y = torch.randn(1, 8, 1, 1)

# Initializing the model
m = Model(8)

# Inputs to the model
z = m(x, other=y)
