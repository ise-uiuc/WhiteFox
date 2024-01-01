
class Model(torch.nn.Module):
    def __init__(self, linear, negative_slope):
        super().__init__()
        self.linear = linear
        self.negative_slope = negative_slope
 
    def get_leaky_relu(self):
        return lambda self: Lambda(lambda x: F.leaky_relu(x, self.negative_slope))
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
linear = torch.nn.Linear(6, 8, bias=False)
negative_slope = 0.01
m = Model(linear, negative_slope)
leaky = m.get_leaky_relu()

# Inputs to the model
x2 = torch.randn(1, 6)
