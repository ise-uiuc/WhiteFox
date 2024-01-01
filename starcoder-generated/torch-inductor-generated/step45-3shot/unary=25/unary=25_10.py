
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = x1.reshape(x1.shape[0], -1)
        v2 = torch.where(v1 > 0, v1, v1 * self.negative_slope)
        return v2.reshape(*x1.shape)

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
