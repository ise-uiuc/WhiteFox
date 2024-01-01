
class Model1(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(torch.nn.Linear(13, 10))

# Inputs to the model
x1 = torch.rand(1, 13)

# Another tensors to the model
other = torch.rand(1, 10)
