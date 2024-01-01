
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 30)
 
    def forward(self, x1):
        v1 = x1.view(10, 20)
        v2 = self.linear(v1)
        v3 = 2.0 - v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
