
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, kwargs['min_value'])
        v3 = torch.clamp_max(v2, kwargs['max_value'])
        return v3

# Initializing the model
m1 = Model().eval()

# Inputs to the model
x1 = torch.randn(1, 3)
