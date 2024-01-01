
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
w = torch.ones(8, 3)
b = torch.empty(8)
m = Model()
with torch.no_grad():
    m.linear.weight = torch.nn.Parameter(w)
    m.linear.bias = torch.nn.Parameter(b)

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(8, 3)
