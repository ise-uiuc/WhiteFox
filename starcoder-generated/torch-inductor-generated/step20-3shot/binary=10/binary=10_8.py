
class Model(torch.nn.Module):
    def __init__(self):
      ...
    
    def forward(self, x):
      ...
        other = torch.randn(3)
        v1 = self.fc(xx)
        v2 = v1 + other 
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
