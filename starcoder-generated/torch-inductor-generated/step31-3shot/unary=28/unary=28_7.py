
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3
 
init = Initializer(min_value=-1, max_value=1)
m = Model()

# Inputs to the model
x1 = torch.randn(10, 8, 16)
