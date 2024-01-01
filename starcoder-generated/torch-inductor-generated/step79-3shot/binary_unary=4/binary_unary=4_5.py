
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3
        
# Initializing the model
other = torch.randn(4)
m = Model(other)

# Inputs to the model
x1 = torch.randn(5, 3)
