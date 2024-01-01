
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
  
    def forward(self, x, other=-3.0):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3, v2
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
