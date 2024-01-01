
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=True)
 
    def forward(self, x: torch.Tensor):
        v = self.linear(x).clamp_min(min_value=min_value).clamp_max(max_value=max_value)
        return v

# Initializing the model
min_value = -1.5
max_value = 5.5
m = Model(min_value, max_value)

# Inputs to the model
x = torch.randn(1, 16)
