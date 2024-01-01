
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(9, 8)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.clamp_min(x2, min_value)
        x4 = torch.clamp_max(x3, max_value)
        return x4

# Initializing the model
m = Model(1, 20)

# Inputs to the model
x1 = torch.randn(1, 9)
