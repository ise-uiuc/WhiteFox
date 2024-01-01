
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=32, out_features=32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = self.negative_slope
        v4 = torch.where(v2, v1, v1 * v3)
        return v4

# Initializing the model
m = Model()

# Assigning the value of the negative slope
m.negative_slope = 0.1

# Inputs to the model
x1 = torch.randn(1, 32)
