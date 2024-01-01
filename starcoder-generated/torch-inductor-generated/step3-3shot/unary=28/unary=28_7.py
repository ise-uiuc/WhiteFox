
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1, max_value=0.2):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        v3 = torch.clamp_max(v2, max=self.max_value)
        return v3

# Initializing the model
# This pattern takes optional arguments. If optional arguments are required, please add additional functions in the sample_app.py without the following function call.
m = Model()

# Model inputs to use
x1 = torch.randn(1, 8)
