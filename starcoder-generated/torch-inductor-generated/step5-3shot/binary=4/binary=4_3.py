
class Model(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear = torch.nn.Linear(dim1, dim2)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
d1 = 10
d2 = 20
m = Model(d1, d2)

# First input tensor to the model
x1 = torch.randn(1, 1, d1)

# Second input tensor to the model
x2 = torch.randn(1, 1, d1)

# Inputs to the model
