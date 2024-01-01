
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 20000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        # The value of 'other', which is negative 2.5e-07
        v2 = -2.5e-07
        v3 = v1 - v2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 20000)
