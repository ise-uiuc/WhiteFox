
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 2)
 
    def forward(self, x1, negative_slope=0.02):
        x1 = x1.flatten(start_dim=1)
        v1 = torch.nn.functional.leaky_relu(self.linear(x1), negative_slope)
        return v1

# Initializing the model with non-default negative_slope
m = Model()

# Inputs to the model
x1 = torch.randn(100, 1024)
