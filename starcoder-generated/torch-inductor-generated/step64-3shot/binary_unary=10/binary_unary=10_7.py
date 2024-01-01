
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16)
 
    def forward(self, x3):
        v1 = self.linear(x3)
        v2 = v1 + 1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model.
x3 = torch.randn(2, 10)
