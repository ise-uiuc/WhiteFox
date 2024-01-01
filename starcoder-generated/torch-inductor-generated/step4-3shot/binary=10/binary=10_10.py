
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 3)
 
    def forward(self, x1):
        h1 = self.linear(x1)
        v1 = h1 + h2
        return v1

# Initializing the model
h2 = torch.randn(1, 3)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7)
