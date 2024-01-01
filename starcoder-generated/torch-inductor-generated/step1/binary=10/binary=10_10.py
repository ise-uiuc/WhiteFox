
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.add(v1, x)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
