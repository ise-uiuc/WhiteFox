
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, other):
        v1 = self.linear(other)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
other = torch.randn(1, 10)
