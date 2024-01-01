
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()
# Input tensor
x1 = torch.randn(1, 3)
# The parameter of the add operation
other = torch.rand(1, 8)
