
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(16, 24)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope # Negative slope will be provided as an input
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.3)

# Inputs to the model
x = torch.randn(1, 16)
