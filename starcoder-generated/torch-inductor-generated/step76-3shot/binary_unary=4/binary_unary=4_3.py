
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(6, 8)
 
        self.other = other # Initialize self.other from the input argument
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(6, 8)) # Supply the other tensor as a keyword argument

# Inputs to the model
x1 = torch.randn(1, 6)
