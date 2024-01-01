
class Model(torch.nn.Module):
    def __init__(self, negative_slope, training):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
        self.negative_slope = negative_slope
        self.training = training
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = torch.where(self.training, v1, v4)
        return v5

# Initializing the model
m = Model(0.01, True)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
