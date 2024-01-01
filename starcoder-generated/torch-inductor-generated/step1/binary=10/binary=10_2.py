
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x, other):
        v1_1 = self.linear(x)
        v1_2 = v1_1 + other
        return v1_2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 8, 64, 64)
