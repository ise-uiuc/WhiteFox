
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + other
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 16)
other = torch.randn(5, 32)
