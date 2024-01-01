
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1, min_value=0.1, max_value=-0.1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=min_value)
        v3 = torch.clamp(v2, max=max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)

# Min & Max value 
min_value = 0.0
max_value = -1.0
