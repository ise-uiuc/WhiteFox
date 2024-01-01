
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features = 3, out_features = 11, bias = True)
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = torch.clamp_min(v0, min_value = -0.2)
        v2 = torch.clamp_max(v1, max_value = 7.04)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
