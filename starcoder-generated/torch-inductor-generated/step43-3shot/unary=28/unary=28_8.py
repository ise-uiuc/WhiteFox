
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 3)
 
    def forward(self, x1, min_value=-3, max_value=3):
        v1 = self.linear(x1)
        v2 = v1.clamp(min=min_value)
        v3 = v2.clamp(max=max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
