
class Model(torch.nn.Module):
    def __init__(self, min_value=0.31622776601683794, max_value=2.718281828459045):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=min_value)
        v3 = torch.clamp_max(v2, max_value=max_value)
        return v3

# Initializing parameters to pass to the model
min_value=0.31622776601683794
max_value=2.718281828459045

# Initializing the model
m = Model(
    min_value,
    max_value,
)

# Inputs to the model
x1 = torch.randn(1, 4)
