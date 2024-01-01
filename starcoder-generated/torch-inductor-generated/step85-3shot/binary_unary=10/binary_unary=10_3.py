
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = 100
        n = 80
        self.linear = torch.nn.Linear(m, n)
        self.other = torch.nn.Parameter(torch.rand(n))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100)
