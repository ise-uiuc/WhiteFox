
class Model(torch.nn.Module):
    def __init__(self):
       self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x0):
       v0 = self.linear(x0)
       v1 = torch.sigmoid(v0)
       return v1

# Initializing the model
m = Model()

# Input to the model
x0 = torch.randn(20, 5)
