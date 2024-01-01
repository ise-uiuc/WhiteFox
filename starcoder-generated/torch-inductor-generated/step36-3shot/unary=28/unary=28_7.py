
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3


# Initializing an instance of Model with min_value = 0 and max_value = 1
m = Model(0, 1)

# Input to the model
x1 = torch.randn(1, 3, 64, 64)
