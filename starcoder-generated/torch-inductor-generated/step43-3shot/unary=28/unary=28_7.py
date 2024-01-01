
class Model(torch.nn.Module):
    def __init__(self, min=0, max=5):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return v3
m = Model(min=-5.0, max=5.0)

# Inputs to the model
x1 = torch.randn(1, 10)
