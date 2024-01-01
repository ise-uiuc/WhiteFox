
class Model(torch.nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
        self.min_value = -max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2

# Initializing the model
m = Model(__init_param__)

# Inputs to the model
x1 = torch.randn(1, 64)
