
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1, other):
        return self.linear(x1) + other
 
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 32)
other = torch.randn(16, 16)
return m(x1, other)

# Inputs to the model
x1 = torch.randn(3, 4, 5, 5)
other = torch.tensor([10])
return torch.sum(x1 * other.item())

