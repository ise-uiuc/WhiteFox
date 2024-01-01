
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = o1 + other
        o3 = torch.nn.functional.relu(o2)
        return o3

# Initializing the model
other = torch.randn(4, 3) # An example for another tensor
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
