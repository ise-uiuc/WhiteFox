
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(20, 30)
 
    def forward(self, x, y):
        v1 = self.lin(x)
        v2 = v1 + y
        return v2

# Initializing the model
m2 = Model()

# Inputs to the model
x = torch.randn(32, 20)
y = torch.randn(32, 30)
__output = m2(x, y)




