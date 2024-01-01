
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(1, 2)
        self.positive = torch.nn.Linear(1, 2)
        self.negative_slope = 0.3
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.leaky_relu(v1, negative_slope=self.negative_slope)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
