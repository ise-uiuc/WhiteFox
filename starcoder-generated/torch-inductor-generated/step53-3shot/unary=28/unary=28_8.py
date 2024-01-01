
class Model(torch.nn.Module):
    def __init__(self, min_value=0.24243659, max_value=-0.15193957):
        super().__init__()
        self.linear = torch.nn.Linear(100, 256)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=min_value)
        v3 = torch.clamp_max(v2, max=max_value)
        return v3

# Initializing the model
m = Model(min_value=0.16591944, max_value=.01278527)

# Inputs to the model
x1 = torch.randn(1, 100)
