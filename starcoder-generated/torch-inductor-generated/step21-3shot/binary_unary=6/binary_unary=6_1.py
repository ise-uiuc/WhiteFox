
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.other = torch.nn.Parameter(torch.randn(100).view(1, 100))
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 - self.other.squeeze()
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 100)
