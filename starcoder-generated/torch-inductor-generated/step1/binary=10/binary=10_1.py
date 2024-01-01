
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x, y):
        v1 = self.linear(x)
        v2 = v1 + y
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 8)
y = torch.randn(3, 8)
