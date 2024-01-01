.

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x):
        b = self.linear(x)
        c = b + x
        d = torch.relu(c)
        return d

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 100)
