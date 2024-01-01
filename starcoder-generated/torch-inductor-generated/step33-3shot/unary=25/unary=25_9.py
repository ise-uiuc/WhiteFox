
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=0.5) # If m.negative_slope == 0, then the model is the same as the one in Model 1. If m.negative_slope!= 0, then this is the new model.

# Inputs to the model
x1 = torch.randn(5, 32)
