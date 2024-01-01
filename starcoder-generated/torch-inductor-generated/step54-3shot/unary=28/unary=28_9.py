
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = torch.clamp_min(v2, min=-6.0)
        v4 = torch.clamp_max(v3, max=6.0)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(2, 784)
