
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.other = torch.randn(32)
 
    def forward(self, x1):
        h1 = self.linear(x1)
        h2 = h1 - self.other
        return h2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
