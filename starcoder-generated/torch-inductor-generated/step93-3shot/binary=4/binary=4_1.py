
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, linear_1):
        v1 = self.linear(x1)
        v2 = v1 + linear_1()
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
linear_1 = torch.nn.Linear(3, 8)
m.linear = linear_1
