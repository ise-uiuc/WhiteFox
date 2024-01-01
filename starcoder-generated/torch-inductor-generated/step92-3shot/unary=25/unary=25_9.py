
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.conv.weight, self.conv.bias)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model with a negative slope of 0.2
negative_slope = torch.tensor([0.2])
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(4, 32)
