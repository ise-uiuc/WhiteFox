
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(4, 64)
