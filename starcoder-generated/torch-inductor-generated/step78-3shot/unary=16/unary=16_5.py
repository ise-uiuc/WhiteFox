
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 32)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 32)
