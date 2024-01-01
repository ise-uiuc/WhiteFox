
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        t = torch.randn(1, 3)
        v = self.linear(v1, other=t)
        v2 = torch.relu(v)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 3)
