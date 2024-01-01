
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=1)
 
    def forward(self, x1, other):
        v1 = self.linear(x3)
        v3 = v1 + other
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 128)
x2 = torch.randn(10, 128)
