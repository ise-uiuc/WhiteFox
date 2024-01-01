
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x, other=None):
        if other is None:
            other = x
        
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(2, 10)
