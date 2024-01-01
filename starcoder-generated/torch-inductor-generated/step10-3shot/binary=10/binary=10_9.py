
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
 
    def forward(self, x, other):
        x = self.linear(x)
        x = torch.add(x, other)
        return x

# Initializing the model
a = A()

# Inputs to the model
x = torch.randn(1, 3)
other = torch.rand(1, 2)
