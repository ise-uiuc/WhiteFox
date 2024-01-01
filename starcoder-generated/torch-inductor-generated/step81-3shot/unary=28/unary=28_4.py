
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.min_value = min_value
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, 0.7758959098339039)
        return v3

# Initializing the model
m = Model(0.04188456921332358)

# Inputs to the model
x2 = torch.randn(1, 10)
