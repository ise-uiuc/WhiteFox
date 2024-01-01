
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=torch.zeros([])):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model 
m = Model()

# Input tensor to the model
x = torch.randn(1, 3)

# Other constant added to the model
o = torch.randn(1, 8)

# Inputs to the model
