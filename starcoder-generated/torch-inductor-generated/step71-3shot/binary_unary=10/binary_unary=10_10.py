
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.linear.weight = torch.nn.Parameter(torch.randn(8, 3))
        self.linear.bias = torch.nn.Parameter(torch.randn(8))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.b1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()
m.b1 = torch.randn(8)

# Inputs to the model
x1 = torch.randn(1, 3)
