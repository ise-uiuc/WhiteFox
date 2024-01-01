
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(10, 10)
        self.linear_2 = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = self.linear_2(v1)
        return v2 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 16, 17)
