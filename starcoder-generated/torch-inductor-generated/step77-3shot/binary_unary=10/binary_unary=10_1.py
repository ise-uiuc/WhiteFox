
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 8)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + other
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
