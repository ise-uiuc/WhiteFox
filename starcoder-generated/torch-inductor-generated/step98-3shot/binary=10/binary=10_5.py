
class Model(torch.nn.Module):
    def __init__(self, other_value):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.other = other_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model with the specified weights
other_value = torch.randn(4, 2)
m = Model(other_value)

# Inputs to the model
x1 = torch.randn(1, 2)
