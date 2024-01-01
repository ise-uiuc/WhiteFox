
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 1)
        return v3

# Initializing the model
m = Model()

# Inputs and keyword arguments to the model
x = torch.randn(1, 16)
min_value = torch.randn(1, 16)
max_value = torch.randn(1, 16)
