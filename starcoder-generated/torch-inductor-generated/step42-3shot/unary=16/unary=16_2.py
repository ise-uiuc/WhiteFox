
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x2):
        v0 = self.linear(x2)
        v1 = torch.relu(v0)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
