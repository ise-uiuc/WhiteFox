
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64, bias=False)
 
    def forward(self, x1):
        z1 = self.linear(x1)
        v1 = z1 - 3.14
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
