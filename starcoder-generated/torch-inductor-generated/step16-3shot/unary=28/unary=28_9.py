
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1, min_value=0, max_value=0):
        v1 = self.linear(x1)
        return min(max(v1, min_value), max_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
