
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        negative_slope = 0.05
 
    def forward(self, x):
        x = x * negative_slope
        x = torch.where(x > 0, x, x * negative_slope)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x_init = (torch.randn(3, 8) - 1.0) * 10
