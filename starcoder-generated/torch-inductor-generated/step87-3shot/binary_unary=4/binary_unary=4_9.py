
class Model(torch.nn.Module):
    def __init__(self, other): # `other` is a keyword argument
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.randn(1, 32))

# Inputs to the model
x1 = torch.randn(1, 32)
