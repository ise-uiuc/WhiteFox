
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(64, 32)
 
    def forward(self, x1, other):
        v1 = self.l(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
x = torch.randn(1, 64)
other=torch.randn(1, 32)
m = Model()

# Inputs to the model
