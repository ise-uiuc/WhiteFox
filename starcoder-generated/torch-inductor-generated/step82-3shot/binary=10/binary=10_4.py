
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(2, size)
 
    def forward(self, x1, y1):
        v1 = self.linear(x1)
        v2 = v1 + y1
        return v2

# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 2)
y1 = torch.randn(1, 3)
