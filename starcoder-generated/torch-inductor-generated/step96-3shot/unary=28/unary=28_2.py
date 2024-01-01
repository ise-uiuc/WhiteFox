
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        W = torch.empty(3, 3, requires_grad=True)
        torch.nn.init.uniform_(W)
        self.linear = torch.nn.Linear(3, 3, weight=W, bias=None)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model(min_value=0.0, max_value=3.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
