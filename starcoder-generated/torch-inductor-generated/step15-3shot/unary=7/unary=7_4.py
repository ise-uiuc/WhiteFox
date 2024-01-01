
def relu(x, min=0., max=6.):
    return torch.clamp(min=min, max=max, input=x)
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * relu(x=v1, min=0.0, max=6.0)
        return v2 / 6
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
