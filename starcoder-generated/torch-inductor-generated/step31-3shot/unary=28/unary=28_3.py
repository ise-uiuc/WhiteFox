
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(18, 9)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, -1000)
        v3 = torch.clamp_max(v2, 1000)
        return v3

# Initializing the model
m = Model()

# The minimum value is provided as a keyword argument
min_value = -1000

# The maximum value is provided as a keyword argument
max_value = 1000

# The input tensor is generated randomly
x1 = torch.randn(1, 18)

# The generated model is tested against some random inputs
