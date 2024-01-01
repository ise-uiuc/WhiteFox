
class Model(torch.nn.Module):
    def __init__(self, min_n, max_n):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 1000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_n)
        v3 = torch.clamp_max(v2, max_n)
        return v3

# Initializing the model
m = Model(100, 10000)

# Inputs to the model
x1 = torch.randn(1000)
min_n = 0.1
max_n = 0.5
