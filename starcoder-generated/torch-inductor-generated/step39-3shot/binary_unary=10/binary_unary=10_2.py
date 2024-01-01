
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.fc2 = torch.nn.Linear(4, 8)
 
    def forward(self, x1, y1):
        v1 = self.linear(x1)
        v2 = v1 + y1
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
y1 = torch.randn(1, 2)
