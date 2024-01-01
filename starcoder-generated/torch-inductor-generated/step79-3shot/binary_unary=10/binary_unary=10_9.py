
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, v1):
        v5 = self.linear(v1)
        v7 = v5 + v2
        v8 = torch.nn.functional.relu(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 5)
v2 = torch.randn(1, 5)
