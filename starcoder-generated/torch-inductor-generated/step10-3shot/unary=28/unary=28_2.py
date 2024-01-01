
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.min_val = min_value
        self.max_val = max_value
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.clamp(v1, min=self.min_val)
        v3 = torch.clamp(v2, max=self.max_val)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model(min_value=-0.5, max_value=5)

# Inputs to the model
x1 = torch.randn(1, 128)
