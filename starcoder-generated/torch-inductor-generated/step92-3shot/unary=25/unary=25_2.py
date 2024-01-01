
if negative_slope is None:
    negative_slope = 1.0 / math.sqrt(5)

class Model(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=20, out_features=10)
        self.activation = activation
 
    def forward(self, x0):
        v2 = self.linear(x0)
        v3 = v2 > 0
        v5 = self.activation(v2, v3)
        return v5

# Initializing the model
m = Model(functional.leaky_relu)

# Inputs to the model
x0 = torch.randn(1, 20)
