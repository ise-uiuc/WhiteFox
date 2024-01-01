
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.ones(20, 25))
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        return v2


# Initializing the model
m = Model(0.1, 0.2)

# Inputs to the model
x1 = torch.randn(1, 25)
