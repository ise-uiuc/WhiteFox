
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(25088, 10)
        self.other = other
     
    def forward(self, x1):
        return self.linear(x1) + self.other

# Initializing the model
other_tensor = torch.randn(1, 10)
m = Model(other=other_tensor)

# Inputs to the model
x1 = torch.randn(1, 25088)
