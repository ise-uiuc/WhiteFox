
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(42, 3)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
other = torch.randn(2, 3)
m = Model()

# Inputs to the model
x = torch.randn(1, 42)
