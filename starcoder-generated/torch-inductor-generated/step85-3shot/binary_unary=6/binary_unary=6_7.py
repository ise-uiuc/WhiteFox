
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2, v1):
        v2 = self.linear(x1)
        v3 = v2 - x2
        v4 = F.relu(v3)
        v5 = v4 + v1
        return v5
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(8, 10)
x2 = torch.randn(8, 10)
v1 = torch.randn(8, 10, dtype=torch.float64)
