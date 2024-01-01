
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        self.negative_slope = negative_slope
        super().__init__()
        self.linear = torch.nn.Linear(1000, 500)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        is_positive = v1 > 0
        v2 = v1 * self.negative_slope
        v3 = torch.where(is_positive, v1, v2)
        return v3
    
# Initializing the model
negative_slope = 0.1
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(1, 1000)
