
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
 
    def forward(self, x):
        p1 = self.linear(x)
        p2 = p1 - other
        p3 = torch.relu(p2)
        return p3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
