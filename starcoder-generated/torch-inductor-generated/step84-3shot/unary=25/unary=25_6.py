
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v2 = self.linear(x1) > 0
        v3 = self.linear(x1) * self.negative_slope
        v4 = torch.where(v2, self.linear(x1), v3)
        return v4

# Initializing the model
m = Model(negative_slope=0.3)

# Inputs to the model
x1 = torch.randn(1, 3)
