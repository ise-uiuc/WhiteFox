
class Model(torch.nn.Module):
    def __init__(self, min_value = 0.5, max_value = 0.8):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        b, c, _, _ = x1.shape
        x1 = x1.reshape(b, -1)
        v1 = self.linear(x1)
        v2 = v1.clamp(min = self.min_value)
        v3 = v2.clamp(max = self.max_value)
        return v3
min_value = 0.5
max_value = 0.8

# Initializing the model
m = Model(min_value = min_value, max_value = max_value)

# Inputs to the model
x1 = torch.randn(1, 6, 10, 8)
