
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 8)
v = torch.randn(10, 12)
