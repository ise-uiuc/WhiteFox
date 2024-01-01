
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(20, 30)
 
    def forward(self, x, other):
        v1 = self.fc1(x)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 20)
