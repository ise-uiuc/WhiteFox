
class Model(torch.nn.Module):
    def __init__(self, linear, other):
        super().__init__()
        self.linear = linear
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        result = v1 + self.other
        return result

linear = torch.nn.Linear(3, 3)
other = torch.randn(1, 3)

# Initializing the model
m = Model(linear, other)

# Input to the model
x1 = torch.randn(1,3)

# Outputs of the model. They are same with inputs.
