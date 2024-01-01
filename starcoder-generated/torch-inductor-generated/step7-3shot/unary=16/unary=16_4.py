
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v1 = torch.relu(v2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(32, 64)
