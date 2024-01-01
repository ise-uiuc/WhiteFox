
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = v1 + self.other
        return v3

# Initializing the model
m = Model(other=torch.tensor([[1.1]]))

# Inputs to the model
x1 = torch.randn(1, 1)
