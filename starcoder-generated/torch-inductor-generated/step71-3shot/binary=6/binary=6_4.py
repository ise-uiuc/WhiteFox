
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 1000)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(100, 256)

# Expected value of feature pyramid
