
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_ = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v2 = self.linear_(x1)
        v3 = v2 - 1
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
