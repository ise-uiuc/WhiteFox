
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9, 10)
 
    def forward(self, x, t):
        v1 = self.linear(x)
        v2 = v1 + t
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 9)
t1 = torch.randn(8, 10)
