
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(1200, 84)
 
    def forward(self, x1):
        v1 = self.flatten(x1)
        v2 = self.linear(v1)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 32, 4)
