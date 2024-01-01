
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        # If negative_slope is negative, a leaky ReLU function will be used, 
        # and the absolute value of negative_slope will be used as the slope value of the Leaky ReLU function. 
        if negative_slope < 0:
            self.negative_slope = -negative_slope
            self.leaky_relu = torch.nn.LeakyReLU(0, True)
        else:
            self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m1 = Model(-0.5)
m2 = Model(0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
__output1__ = m1(x1)
__output2__ = m2(x1)

