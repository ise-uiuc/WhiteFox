
class Model(torch.nn.Module):
    def __init__(self, min_value=-5.0, max_value=5.0):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=self.min_value, max=self.max_value)
        v3 = torch.clamp(v2, min=self.min_value, max=self.max_value)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
