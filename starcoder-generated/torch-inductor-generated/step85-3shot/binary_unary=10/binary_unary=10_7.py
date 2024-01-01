
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(42, 42)
        self.linear2 = torch.nn.Linear(42, 26)
 
    def forward(self, x):
        h1 = self.linear1(x)
        h2 = self.linear2(h1)
        z = h2 + h1
        return z

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 42)
