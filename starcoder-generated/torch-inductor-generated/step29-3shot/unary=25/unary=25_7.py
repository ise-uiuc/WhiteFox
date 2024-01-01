
class Model(torch.nn.Module):
    def __init__(self,negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)

    def forward(self, x1,_negative_slope):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = -_negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initialize the parameters of the model
negative_slope = 0.1

# Initialize the model with the parameters
m = Model(negative_slope)

# Inputs to the model
x1 = torch.randn(2, 1, 8)
