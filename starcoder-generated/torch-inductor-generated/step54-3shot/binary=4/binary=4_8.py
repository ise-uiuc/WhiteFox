
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(2,3)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.add(v1, self.other)
        return v2

# Initializing a randomly-initialized tensor to be added to the input tensor for each forward pass of the model
other = torch.randn(3,2)

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 2)
