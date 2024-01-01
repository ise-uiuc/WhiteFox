
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3*8*8, 3*8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - 0.01
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3*8*8)
