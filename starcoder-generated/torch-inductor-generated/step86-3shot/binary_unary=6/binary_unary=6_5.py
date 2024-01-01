
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        v1 = t1 - x2
        v2 = self.activation(v1)
        return v2
 
def relu(x1):
    v1 = x1.clamp_min(0.)
    return v1
 
# Initializing the model
other = torch.randn(8)
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = other
