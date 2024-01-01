
class Model(torch.nn.Module):
    def __init__(self, extra):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + extra
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(extra=torch.randn(5, 3))

# Inputs to the model
x1 = torch.randn(6, 3)
