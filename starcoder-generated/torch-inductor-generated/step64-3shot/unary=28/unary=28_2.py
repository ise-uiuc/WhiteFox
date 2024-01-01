
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x):
        v1 = self.linear(x)
        return torch.nn.functional.clamp(torch.nn.functional.clamp(v1, min=self.min), max=self.max)

# Initializing the model
m = Model()
m.min = min
m.max = max

# Inputs to the model
x = torch.randn(10, 5)
