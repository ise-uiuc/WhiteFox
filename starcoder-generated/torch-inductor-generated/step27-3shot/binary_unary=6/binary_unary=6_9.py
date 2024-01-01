
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 2)

# Setting'self.linear.weight' and'self.linear.bias'
m.linear.weight = torch.nn.Parameter(torch.randn(2, 2))
m.linear.bias = torch.nn.Parameter(torch.randn(2))

