
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = torch.relu(v0)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
