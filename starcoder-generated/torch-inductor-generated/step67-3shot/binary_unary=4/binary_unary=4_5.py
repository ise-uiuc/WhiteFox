
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, bias=False)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return torch.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3)
other = torch.randn(4, 5, 3)
