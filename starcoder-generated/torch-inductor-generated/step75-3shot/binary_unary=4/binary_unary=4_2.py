
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.other)
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.randn(3, 4))

# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
