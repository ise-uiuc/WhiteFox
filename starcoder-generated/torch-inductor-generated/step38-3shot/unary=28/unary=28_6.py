
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, min, max):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=min)
        v3 = torch.clamp(v2, max=max)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
min = 0.0
max = 2.5
