
class Model(torch.nn.Module):
    def __init__(self, negative_slope=None):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, weight=torch.randn(64, 64), bias=torch.randn(64))
        if self.negative_slope is None:
            v2 = v1 > 0
        else:
            v2 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v2)
        return v4
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
