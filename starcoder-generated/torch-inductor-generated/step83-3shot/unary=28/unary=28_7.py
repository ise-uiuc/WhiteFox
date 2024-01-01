
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=0.4):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = F.clamp(v1, min=self.min_value)
        v3 = F.clamp(v2, max=self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
