
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1).clamp(min=0, max=6)
        return v1/6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
