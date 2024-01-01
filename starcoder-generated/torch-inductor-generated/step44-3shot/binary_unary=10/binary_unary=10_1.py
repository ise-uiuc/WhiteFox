
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = torch.nn.Parameter(other)
 
    def forward(self, x1):
        v1 = x1
        v2 = torch.matmul(v1, self.other)
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(3, 4))

# Inputs to the model
x1 = torch.randn(1, 3)
