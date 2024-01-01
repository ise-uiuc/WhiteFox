
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x, other=2.0):
        v = self.linear(x)
        s = v - other
        return s 

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
