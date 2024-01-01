
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
 
    def forward(self, x, min_value=None, max_value=None):
        v1 = self.linear1(x)
        return v1.clamp(min=min_value, max=max_value)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
