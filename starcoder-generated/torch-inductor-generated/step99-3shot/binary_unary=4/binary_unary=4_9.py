
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16, bias=True)
 
    def forward(self, x1, other=0):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
