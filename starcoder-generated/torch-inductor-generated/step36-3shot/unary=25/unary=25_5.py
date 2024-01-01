
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.leaky_relu.negative_slope
        v4 = torch.where(v2, v1, v3)
        return self.leaky_relu(v4)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2)
