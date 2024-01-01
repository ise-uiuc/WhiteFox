
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(256, 512) # Define a linear transformation with input dimension 256 and output dimension 512
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        v3 = torch.clamp_max(v2, max=self.max_value)
        return v3

# Initializing the model
m = Model(min_value=0, max_value=16)

# Inputs to the model
x1 = torch.randn(1, 256)
