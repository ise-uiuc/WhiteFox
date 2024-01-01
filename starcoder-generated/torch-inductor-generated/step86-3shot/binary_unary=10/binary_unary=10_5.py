
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = self.linear(x2)
        v3 = v2 + x2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 2)
