
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight_shape = (8, 3, 5, 5)
        self.linear = nn.Linear(*weight_shape)
        self.min_value, self.max_value = -1, 1
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.min_value, None)
        v3 = torch.clamp(v2, None, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
