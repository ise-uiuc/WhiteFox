
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        negative_slope = self.negative_slope
        v2 = (v1 > 0)
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4


# Initializing the model and setting the negative slope
negative_slope = 0.3701171878490448
m = Model(negative_slope)


# Inputs to the model
x1 = torch.randn(1, 3)
