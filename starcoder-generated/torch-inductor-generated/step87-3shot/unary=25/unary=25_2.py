
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
 
# Initializing the model
m = Model(-1)

# Inputs to the model
x = torch.randn(1, 10)
