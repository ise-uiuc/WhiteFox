
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = torch.clamp(v2, self.min_value, self.max_value)
        return v3

min_value = 0.5
max_value = 1.5
# Initializing the model
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
