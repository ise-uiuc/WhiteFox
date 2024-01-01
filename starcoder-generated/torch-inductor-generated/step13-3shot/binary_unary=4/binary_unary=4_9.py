
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 4)
 
    def forward(self, __input__, t):
        v1 = self.linear(__input__)
        v2 = v1 + t
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
t = torch.randn(1, 4)

