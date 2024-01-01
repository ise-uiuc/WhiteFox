
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2)
 
    def forward(self, x1, min_value=-20, max_value=5):
        v2 = torch.clamp_min(self.linear(x1), min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(1, 16)
