
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-1)
        v3 = torch.clamp_max(v2, max_value=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
