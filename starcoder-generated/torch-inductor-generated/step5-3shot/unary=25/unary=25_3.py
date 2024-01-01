
class Model(torch.nn.Module):
    def __init__(self, negative_slope, negative_slope_tensor, input_shape, output_shape):
        super().__init__()
        self.linear = torch.nn.Linear(input_shape[-1], output_shape[-1])
        self.negative_slope = negative_slope
        self.negative_slope_tensor = negative_slope_tensor
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
input_shape = (3,)
negative_slope = 0.01
output_shape = (3,)
m = Model(negative_slope, negative_slope_tensor, input_shape, output_shape)

# Inputs to the model
x1 = torch.randn(1, input_shape[-1])
