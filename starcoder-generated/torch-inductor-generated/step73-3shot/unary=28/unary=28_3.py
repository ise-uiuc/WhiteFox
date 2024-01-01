
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.l = torch.nn.Linear(8, 4)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v1 = self.l(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
# min_value = 0
# max_value = 3.2
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
