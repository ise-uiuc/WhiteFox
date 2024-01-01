
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(96, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        # We want to compute negative_slope * v1 for elements that
        # v1 > 0. Negative slope is 0.2
        v2 = v1 > 0
        v3 = v1 * 0.2
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
# We create an input tensor x1 where values in the channel dimension
# alternate between [-10.0, -3.0,...], [10.0, 3.0,...]
x1 = torch.zeros(1, 96, 1, 1)
x1[:, ::2] = -10.0
x1[:, 1::2] = 10.0
