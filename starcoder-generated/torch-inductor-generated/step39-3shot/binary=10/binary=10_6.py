
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x2, o1):
        v1 = self.linear(x2)
        v2 = v1 + o1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
o1 = torch.randn(1, 8)
