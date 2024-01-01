
class Model(torch.nn.Module):
    def __init__(self, i, j, k, p0):
        super().__init__()
        self.linear = torch.nn.Linear(i, j, p0)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
p0 = 1
m = Model(1000, 1, p0, True)

# Inputs to the model
x1 = torch.randn(1, 8, 8)
x2 = torch.randn(1, 8, 8)
