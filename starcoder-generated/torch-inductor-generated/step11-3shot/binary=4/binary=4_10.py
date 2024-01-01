
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 64)
 
    def forward(self, v, x):
        v1 = self.linear1(v)
        v2 = v1 + x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
v  = torch.randn(1, 32)
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
 
# Case 1
