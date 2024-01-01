
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()
m.other = torch.randn(10, 10)

# Inputs to the model
x = torch.randn(10, 5)
