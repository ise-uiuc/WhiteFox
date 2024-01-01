
class Model(torch.nn.Module):
    def __init__(self, min_value=0.01, max_value=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(128, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min_value)
        v3 = torch.clamp_max(v2, max_value=self.max_value)
        return v3

# Initializing the model
m = Model()
m.min_value = 0.01
m.max_value = 0.5

# Inputs to the model
x1 = torch.randn(1, 128)
