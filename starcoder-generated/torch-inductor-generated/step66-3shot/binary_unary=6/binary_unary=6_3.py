
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.randn(50, 100)
        self.linear = torch.nn.Linear(100, 50, weights)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v2 = v3 - OTHER
        v4 = F.relu(v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 100)
