
class ClampMinModel(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.min_value = min_value
    
    def forward(self, x_input):
        if self.training:
            v1 = torch.ones_like(x_input)
        else:
            v1 = x_input
        v2 = v1 * self.min_value
        v3 = torch.clamp(v1, min=self.min_value)
        return v3

# Initializing the model
m = ClampMinModel(0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
