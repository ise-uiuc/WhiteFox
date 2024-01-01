
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1, x2, x3):
        v2 = x2 + x2
        v3 = x3 + x3
        v4 = v2 + v3
        v5 = self.linear(x1)
        v6 = v4 + v5
        return v6
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 3)
