
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        x1 = x1.flatten(1)
        v2 = self.linear(x1)
        v3 = v2 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 32, 32)
