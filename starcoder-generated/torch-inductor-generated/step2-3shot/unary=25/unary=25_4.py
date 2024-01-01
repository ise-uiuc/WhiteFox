
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope  # Negative slope to Leaky ReLU used during conversion of the linear ReLU to the custom ReLU type
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.rand(12, 3), bias=None)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(torch.nn.functional.relu(x1) == 0, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=-0.01)

# Inputs to the model
x1 = torch.randn(1, 12, 3)
