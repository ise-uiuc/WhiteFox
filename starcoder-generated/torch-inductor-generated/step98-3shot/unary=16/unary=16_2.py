
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(n, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model(n=64)

# Inputs to the model
x1 = torch.randn(2, 64)
