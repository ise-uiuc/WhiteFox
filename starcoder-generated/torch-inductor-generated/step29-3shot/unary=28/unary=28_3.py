
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1280, 1280)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 0.8)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1280)
min_value = torch.full((1, ), -10, out=torch.Tensor())
max_value = torch.full((1, ), 10, out=torch.Tensor())
