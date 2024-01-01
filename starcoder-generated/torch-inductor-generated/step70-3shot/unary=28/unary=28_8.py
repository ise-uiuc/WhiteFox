
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = __torch__.torch.nn.functional.linear(x1, None)
        v2 = __torch__.torch.clamp_min(v1, self.min_value)
        v3 = __torch__.torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(-1.85126040678, 1.491218941a)

# Inputs to the model
x1 = torch.randn(1, 5, 5)
