
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)
 
    def forward(self, x1, t1):
        v1 = self.fc(x1)
        v = v1 + t1
        v2 = torch.nn.functional.relu(v)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
t1 = torch.randn(1, 1)
