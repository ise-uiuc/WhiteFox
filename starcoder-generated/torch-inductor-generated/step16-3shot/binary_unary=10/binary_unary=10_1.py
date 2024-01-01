
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1, x2=None):
        v1 = self.linear(x1)
        v2 = v1
        if x2 is not None:
            v2 = x2 + v1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = TestModel()

# Inputs to the model
x1 = torch.randn(1, 3, 6)
x2 = torch.randn(1, 3, 6)
