
class Model(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.linear = torch.nn.Linear(10, 11)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min if min else torch.min(v1))
        v3 = torch.clamp_max(v2, max if max else torch.max(v2))
        return v3

# Initializing the model
model = Model(min=-0.1, max=0.5)

# Inputs to the model
x1 = torch.rand(1, 10)
