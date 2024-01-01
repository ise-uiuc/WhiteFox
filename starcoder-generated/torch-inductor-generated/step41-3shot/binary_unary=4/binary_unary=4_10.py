
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v3

# Initializing the new model
m = Model(torch.rand(64))

# Initializing the input to the model
x1 = torch.rand(1, 64)

# Input to the model
