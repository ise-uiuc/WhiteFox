
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 10)
 
    def forward(self, x):
        z = self.linear(x)
        return torch.where(z > 0, z, z * negative_slope)

# Initializing the model
negative_slope = 0.2
m = Model(negative_slope)

# Inputs to the model
x = torch.randn(1, 1)
