
class Model(torch.nn.Module):
    def __init__(self, min_value=-10, max_value=10):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v0 = self._reshape_1(x1)
        v1 = self.linear(x1)
        v2 = v1.clamp(min=self.min_value)
        v3 = v2.clamp(max=self.max_value)
        return v3
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
