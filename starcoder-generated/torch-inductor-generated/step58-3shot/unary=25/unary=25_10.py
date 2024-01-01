
negative_slope = 0.25

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if hasattr(torch.nn.LeakyReLU, '__constants__'):
            self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)
 
    def forward(self, x1):
        v1 = self.leaky_relu(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model, can use random tensor but should not be constant
x1 = torch.randn(1, 3, 64, 64)
