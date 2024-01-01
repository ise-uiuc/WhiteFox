
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.other = torch.nn.Parameter(torch.randn(5))
 
    def forward(self, x):
        v = self.linear(x)
        v = v - self.other
        v = torch.relu(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
