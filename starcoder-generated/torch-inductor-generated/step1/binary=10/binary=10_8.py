
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x, y):
        v1 = x * 1
        v2 = self.linear(v1)
        v3 = v2 + y
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 16)
y = torch.randn(8, 16)
