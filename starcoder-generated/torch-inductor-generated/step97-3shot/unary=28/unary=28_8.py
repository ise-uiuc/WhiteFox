
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, min_value=0.9, max_value=1.1):
        v1 = self.linear(x1)
        v2 = torch.clamp(x=v1, min=min_value, max=max_value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
