
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(n, 100)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
n = 512 
m = Model()

# Inputs to the model
x1 = torch.randn(1, n)
x2 = torch.randn(1, n)
