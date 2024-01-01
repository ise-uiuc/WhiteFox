
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5, bias=False)
        self.m = min_value
        self.M = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.m, self.M)
        return v2

# Initializing the model
m = Model(min_value=0, max_value=100)

# Input to the model
__x1 = torch.randn(512, 4)
