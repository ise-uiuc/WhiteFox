
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 512)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2


# Initializing the tensors
x1 = torch.randn(1, 1024)
other = torch.randn(1, 512)

# Initializing the model
m = Model(other=other)

# Inputs to the model
