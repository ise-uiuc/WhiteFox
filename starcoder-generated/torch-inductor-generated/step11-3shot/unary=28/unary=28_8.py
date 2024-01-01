
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 10)
 
    def forward(self, x1, min_value=-2, max_value=2):
        v1 = self.linear(x1)
        v2 = torch.clamp_max(v1, max_value)
        v3 = torch.clamp_min(v2, min_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
