
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2.relu()
        return v3

# Initializing the model
nm = TestModel()

# Inputs to the model
x2 = torch.randn(1, 16)
