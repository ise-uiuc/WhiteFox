
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
 
    def forward(self, x):
        v1 = self.layer(x)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model(0, 1)

# Inputs to the model
x = torch.randn(1, 784)
