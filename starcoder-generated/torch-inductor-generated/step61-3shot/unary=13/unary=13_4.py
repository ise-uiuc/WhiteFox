
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 32)
        self.fc2 = torch.nn.Linear(32, 10)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.fc2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
