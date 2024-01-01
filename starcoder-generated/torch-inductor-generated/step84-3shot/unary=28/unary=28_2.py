
class Model(torch.nn.Module):
    def __init__(self, min_value=3, max_value=5):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.ones(128))
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=4, max_value=6)

# Inputs to the model
x1 = torch.randn(1, 128)
__output__  = m(x1)

