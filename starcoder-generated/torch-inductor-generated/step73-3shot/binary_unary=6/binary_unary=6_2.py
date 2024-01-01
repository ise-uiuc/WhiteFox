
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(30, 10)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - 5
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30)
