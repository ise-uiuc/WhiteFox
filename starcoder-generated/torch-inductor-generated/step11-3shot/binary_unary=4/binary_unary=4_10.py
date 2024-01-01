
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
 
    def forward(self, x1, t2):
        v1 = self.linear(x1)
        v3 = v1 + t2
        v4 = functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
t2 = torch.tensor(3)
