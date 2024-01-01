
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.min_value = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(self.min_value)
        v3 = v2.clamp(min=-self.min_value)
        return v2

# Initializing the model
min_value = 0.3
m = Model(-min_value)

# Inputs to the model
x1 = torch.randn(1, 8)
