
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.negative_slope = negative_slope
 
    def forward(self, features):
        intermediate = self.linear(features)
        intermediate = intermediate > 0
        return torch.where(intermediate, intermediate, intermediate * self.negative_slope)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
