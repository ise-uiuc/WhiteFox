
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, a, b):
        v1 = self.linear(a)
        v2 = v1 + b
        return v2

# Initializing the model
m = Model()

# Inputs to the model
a = torch.randn(1, 3)
b = torch.randn(1, 3)
