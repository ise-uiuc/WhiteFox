
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=True)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initialising the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 16)
x2 = torch.randn(32, 32)
