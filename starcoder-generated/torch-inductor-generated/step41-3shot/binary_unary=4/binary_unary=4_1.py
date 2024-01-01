
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 5)
        self.linear2 = torch.nn.Linear(5, 6)
 
    def forward(self, x1, other):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
other = torch.randn(1, 5)
