
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        result = torch.add(v1, other)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5)
other = torch.randn(5)
