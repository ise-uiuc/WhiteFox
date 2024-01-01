
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1000)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model with some random values for the second input of the model
m = Model()
other = torch.randn(1000)

# Inputs to the model
x1 = torch.randn(1, 3)
