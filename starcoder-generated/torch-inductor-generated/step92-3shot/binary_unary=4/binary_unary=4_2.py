
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)
 
    def forward(self, v1):
        v2 = self.linear(v1)
        v3 = v2 + x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
v1 = torch.randn(1, 3)
