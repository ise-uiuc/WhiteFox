
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.l = torch.nn.Linear(10, 10)
 
        self.negative_slope = negative_slope
 
    def forward(self, x1, x2):
        v1 = self.l(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
 
negative_slope = 0.1
# Initializing the model
m = Model(negative_slope)
 
# Inputs to the model
x1 = torch.randn(64,10)
x2 = torch.randn(64,10)
